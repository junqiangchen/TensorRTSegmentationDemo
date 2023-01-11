#include "OnnxTensorRTGlandceilModel.h"
#include <string>
#include <fstream> 
#include <sstream>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

OnnxTensorRTModelGlandceil::OnnxTensorRTModelGlandceil(std::string onnx_file_path, std::string engine_file_path, int inputBatch, int inputC, int inputH, int inputW, int outputBatch, int outputC, int outputH, int outputW)
{
	//should rewrite input dims and output dims value

	this->mInputDims.d[0] = inputBatch;
	this->mInputDims.d[1] = inputC;
	this->mInputDims.d[2] = inputH;
	this->mInputDims.d[3] = inputW;
	this->mOutputDims.d[0] = outputBatch;
	this->mOutputDims.d[1] = outputC;
	this->mOutputDims.d[2] = outputH;
	this->mOutputDims.d[3] = outputW;

	std::ifstream f(onnx_file_path.c_str());
	bool fileflag = f.good();
	if (fileflag)
	{
		this->monnx_file_path = onnx_file_path;
		this->mengine_file_path = engine_file_path;
	}
	else
	{
		std::cout << "pleas input valid onnx model file path" << std::endl;
	}

	this->build();
}
OnnxTensorRTModelGlandceil::~OnnxTensorRTModelGlandceil()
{
	if (this->mstream) cudaStreamDestroy(this->mstream);
}
//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool OnnxTensorRTModelGlandceil::build()
{
	std::ifstream f(this->mengine_file_path.c_str());
	bool fileflag = f.good();
	if (fileflag)
	{
		initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
		std::cout << "Loading TensorRT engine from plan file..." << std::endl;
		std::ifstream file(this->mengine_file_path.c_str(), std::ios::in | std::ios::binary);
		file.seekg(0, file.end);
		size_t size = file.tellg();
		file.seekg(0, file.beg);

		auto buffer = std::unique_ptr<char[]>(new char[size]);
		file.read(buffer.get(), size);
		file.close();

		IRuntime* runtime = createInferRuntime(sample::gLogger.getTRTLogger());
		ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.get(), size);
		this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(engine, samplesCommon::InferDeleter());
	}
	else
	{
		auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
		builder->setMaxBatchSize(this->mbatchSize);
		if (!builder)
		{
			return false;
		}

		const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
		if (!network)
		{
			return false;
		}

		auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
		config->setMaxWorkspaceSize(1 << 30);//256M
		if (!config)
		{
			return false;
		}

		auto parser
			= SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
		if (!parser)
		{
			return false;
		}

		auto constructed = constructNetwork(builder, network, config, parser);
		if (!constructed)
		{
			return false;
		}

		// CUDA stream used for profiling by the builder.
		auto profileStream = samplesCommon::makeCudaStream();
		if (!profileStream)
		{
			return false;
		}
		config->setProfileStream(*profileStream);

		SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
		if (!plan)
		{
			return false;
		}

		SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
		if (!runtime)
		{
			return false;
		}

		this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
		if (!this->mEngine)
		{
			return false;
		}
		//save serialize mode to file
		std::ofstream f(this->mengine_file_path, std::ios::out | std::ios::binary);
		f.write(reinterpret_cast<const char*>(plan->data()), plan->size());
		f.close();
	}
	this->prepare();
	return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool OnnxTensorRTModelGlandceil::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
	SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
	SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	auto parsed = parser->parseFromFile(this->monnx_file_path.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
	if (!parsed)
	{
		return false;
	}

	if (this->mfp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (this->mint8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
	}
	samplesCommon::enableDLA(builder.get(), config.get(), this->mdlaCore);
	return true;
}
bool OnnxTensorRTModelGlandceil::prepare()
{
	this->mcontext = SampleUniquePtr<nvinfer1::IExecutionContext>(this->mEngine->createExecutionContext());
	/*this->mcontext->setOptimizationProfileAsync(0, this->mstream);
	cudaStreamCreate(&this->mstream);*/
	if (!this->mcontext)
	{
		return false;
	}
	return true;
}
//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool OnnxTensorRTModelGlandceil::infer(cv::Mat& image, cv::Mat& ouputmask)
{
	samplesCommon::BufferManager mbuffers(this->mEngine);
	// Read the input data into the managed buffers
	if (!this->processInput(image, mbuffers))
	{
		return false;
	}

	// Memcpy from host input buffers to device input buffers
	mbuffers.copyInputToDevice();
	//mbuffers.copyInputToDeviceAsync(this->mstream);
	//bool status = this->mcontext->enqueueV2(mbuffers.getDeviceBindings().data(), this->mstream, nullptr);
	bool status = this->mcontext->executeV2(mbuffers.getDeviceBindings().data());
	if (!status)
	{
		return false;
	}
	//cudaStreamSynchronize(this->mstream);
	// Memcpy from device output buffers to host output buffers
	mbuffers.copyOutputToHost();
	//mbuffers.copyOutputToHostAsync(this->mstream);
	// Verify results
	if (!processOutput(mbuffers, ouputmask))
	{
		return false;
	}
	return true;
}

//!
//! \brief convert cpu input to gpu output,and process input
//!
bool OnnxTensorRTModelGlandceil::processInput(cv::Mat& image, const samplesCommon::BufferManager& buffers)
{
	const int inputBatch = this->mInputDims.d[0];
	const int inputC = this->mInputDims.d[1];
	const int inputH = this->mInputDims.d[2];
	const int inputW = this->mInputDims.d[3];

	cv::Mat image_cv;
	cv::Size dsize = cv::Size(inputH, inputW);
	cv::resize(image, image_cv, dsize, 0, 0, cv::INTER_LINEAR);
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(this->minputTensorNames[0]));
	zscorenormal(image_cv, hostDataBuffer);
	//std::cout << "inputBatch:" << inputBatch << "," << "inputH:" << inputH << "," << "inputW:" << inputW << "," << "inputC:" << inputC << std::endl;
	return true;
}
//!
//! \brief convert gpu output to cpu output,and process output
//!
//! \return whether the classification output matches expectations
//!
bool OnnxTensorRTModelGlandceil::processOutput(const samplesCommon::BufferManager& buffers, cv::Mat& ouputmask)
{
	//should rewrite output dims
	const int outputBatch = this->mOutputDims.d[0];
	const int outputC = this->mOutputDims.d[1];
	const int outputH = this->mOutputDims.d[2];
	const int outputW = this->mOutputDims.d[3];
	//std::cout << "outputBatch:" << outputBatch << "," << "outputC:" << outputC << "," << "outputH:" << outputH << "outputW:" << outputW << std::endl;
	float* output = static_cast<float*>(buffers.getHostBuffer(this->moutputTensorNames[1]));
	
	cv::Mat ouputmaskfloat = cv::Mat(this->mOutputDims.d[2], this->mOutputDims.d[3], CV_32FC1);
	int hostid = 0;
	for (int i = 0; i < ouputmaskfloat.rows; ++i)
	{ //获取第i行首像素指针 
		for (int j = 0; j < ouputmaskfloat.cols; ++j)
		{
			ouputmaskfloat.at<float>(i, j) = output[hostid];
			hostid++;
		}
	}
	cv::Size dsize = cv::Size(this->mInputDims.d[2], this->mInputDims.d[3]);
	cv::resize(ouputmaskfloat, ouputmaskfloat, dsize, 0, 0, cv::INTER_NEAREST);
	ouputmaskfloat = ouputmaskfloat * 255.;
	ouputmaskfloat.convertTo(ouputmask, CV_8UC1);
	return true;
}