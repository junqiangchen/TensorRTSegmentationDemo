#include "imageprocess.hpp"

bool zscorenormal(cv::Mat& img, float*& hostDataBuffer)
{
	cv::Mat MeanMat, StddevMat;
	cv::meanStdDev(img, MeanMat, StddevMat);
	double m = MeanMat.at<double>(0, 0);
	double sd = StddevMat.at<double>(0, 0);
	cv::Mat imgfloat;
	img.convertTo(imgfloat, CV_32FC1);
	imgfloat = (imgfloat - m) / sd;
	int hostid = 0;
	for (int i = 0; i < imgfloat.rows; ++i)
	{ //获取第i行首像素指针 
		for (int j = 0; j < imgfloat.cols; ++j)
		{
			hostDataBuffer[hostid] = imgfloat.at<float>(i, j);
			hostid++;
		}
	}
	return true;
}