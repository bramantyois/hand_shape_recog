#pragma once
#ifndef REALSENSE_VIEWER
#define REALSENSE_VIEWER

#include <openCV2/opencv.hpp>
#include <iostream>

#define MAX_DEPTH 10000

inline void calculateHistogram(float* pHistogram, int histogramSize,cv::Mat frame)
{
	const uint16_t * pDepth = (const uint16_t*)frame.data;
	// Calculate the accumulative histogram (the yellow display...)
	memset(pHistogram, 0, histogramSize * sizeof(float));
	int height = frame.size().height;
	int width = frame.size().width;

	unsigned int nNumberOfPoints = 0;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x, ++pDepth)
		{
			if (*pDepth != 0)
			{
				pHistogram[*pDepth]++;
				nNumberOfPoints++;
			}
		}
	}
	for (int nIndex = 1; nIndex<histogramSize; nIndex++)
	{
		pHistogram[nIndex] += pHistogram[nIndex - 1];
	}
	if (nNumberOfPoints)
	{
		for (int nIndex = 1; nIndex<histogramSize; nIndex++)
		{
			pHistogram[nIndex] = (256 * (1.0f - (pHistogram[nIndex] / nNumberOfPoints)));
		}
	}
}

class RealSenseViewer
{
public:
	RealSenseViewer(std::string folderPath);
	~RealSenseViewer();

	bool run();

	cv::Mat getColorMat() { return colorFrame_; }
	cv::Mat getDepthMat() { return depthFrame_; }
	cv::Mat getDepthMat8() { return depthFrame8_; }
private:

	cv::String folderPath_;
	std::vector<cv::String> filenames_;
	std::vector<cv::String> filenamesRGB_;
	std::vector<cv::String> filenamesDepth_;

	int frameWidth_;
	int frameHeight_;

	int indexAll_;
	int indexRGB_;
	int indexDepth_;
	
	cv::Mat colorFrame_;
	cv::Mat depthFrame_;
	cv::Mat depthFrame8_;

	float depthHistogram_[MAX_DEPTH];

	int * tempDepthValues_;
	int tempDepthValuesSize_;

	bool initilized_;
};


#endif // !REALSENSE_VIEWER
