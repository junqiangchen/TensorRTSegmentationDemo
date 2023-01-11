#ifndef ONNXTENSORRTGLANDCEILMODEL_H
#define ONNXTENSORRTGLANDCEILMODEL_H

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "imageprocess.hpp"

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;


//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class OnnxTensorRTModelGlandceil
{
public:
	OnnxTensorRTModelGlandceil(std::string onnx_file_path, std::string engine_file_path, int inputBatch, int inputC, int inputH, int inputW, int outputBatch, int outputC, int outputH, int outputW);
	~OnnxTensorRTModelGlandceil();
	//!
	//! \brief Runs the TensorRT inference engine for this sample,should rewrite
	//!
	bool infer(cv::Mat& image, cv::Mat& ouputmask);
	
private:
	std::string monnx_file_path;//onnx model file path
	std::string mengine_file_path;//tensorrt model file path
	std::vector<std::string> minputTensorNames = { "input" };//depend on network inputs
	std::vector<std::string> moutputTensorNames = { "output0","output1" };//depend on network outputs

	int32_t mbatchSize{ 1 };              //!< Number of inputs in a batch
	int32_t mdlaCore{ -1 };               //!< Specify the DLA core to run network on.
	bool mint8{ false };                  //!< Allow runnning the network in Int8 mode.
	bool mfp16{ false };                  //!< Allow running the network in FP16 mode.

	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

	std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
	// Create RAII buffer manager object
	SampleUniquePtr<nvinfer1::IExecutionContext> mcontext;
	cudaStream_t mstream = nullptr;
	//!
	//! \brief Function builds the network engine
	//!
	bool build();
	//!
	//! \brief Parses an ONNX model for MNIST and creates a TensorRT network or load TensorRT network file
	//!
	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
		SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
		SampleUniquePtr<nvonnxparser::IParser>& parser);
	//!
	//! \brief Function prepare the network engine
	//!
	bool prepare();

	//!
	//! \brief Reads the input  and stores the result in a managed buffer,should rewrite
	//!
	bool processInput(cv::Mat& image, const samplesCommon::BufferManager& buffers);
	//!
	//! \brief Classifies digits and verify result,should rewrite
	//!
	bool processOutput(const samplesCommon::BufferManager& buffers, cv::Mat& ouputmask);
};
#endif