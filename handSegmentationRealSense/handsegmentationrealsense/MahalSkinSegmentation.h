#pragma once
#ifndef MAHAL_SKIN_SEGEMENTATION
#define MAHAL_SKIN_SEGEMENTATION
#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>

struct UCIrvineRGB
{
	cv::Vec3b colorData;
	unsigned char type;
};

struct UCIrvineDataSet
{
	int numOfData;
	int numOfRealData;
	UCIrvineRGB * data;
};

class MahalSkinSegmentation
{
public:
	MahalSkinSegmentation();
	~MahalSkinSegmentation();

	bool fetchData(std::string fileAddress, bool useOnlyRealDataSet);
	
	bool calcVarianceAndMean();

	cv::Mat getMahalanobisDistImg(const cv::Mat sourceImg);

private:
	UCIrvineDataSet dataSet_;
	UCIrvineRGB * rgbData_;

	std::string fileAddress_;

	bool isDataValid_;
	bool isIncovarMeanCalculated_;
	bool useOnlyRealDataSet_;

	cv::Mat invCovarMat_;
	cv::Vec3d meanVec_;
};

#endif