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
	MahalSkinSegmentation(float mahalDistance);
	~MahalSkinSegmentation();

	bool fetchData(std::string fileAddress, bool useOnlyRealDataSet);	
	bool calcVarianceAndMean();

	void setMahalDistance(float newDistance);

	cv::Mat getMahalanobisDistImg(const cv::Mat sourceImg);
	cv::Mat getMask(cv::Mat sourceImg);

private:
	UCIrvineDataSet dataSet_;
	UCIrvineRGB * rgbData_;

	std::string fileAddress_;

	bool isDataValid_;
	bool isIncovarMeanCalculated_;
	bool useOnlyRealDataSet_;
	
	float mahalDistance_;

	cv::Mat invCovarMat_;
	cv::Vec3d meanVec_;
};

#endif