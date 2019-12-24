#pragma once
#ifndef REALSENSE_VIEWER
#define REALSENSE_VIEWER

#include <openCV2/opencv.hpp>
#include <iostream>

#define MAX_DEPTH 10000

class RealSenseViewer
{
public:
	RealSenseViewer(std::string folderPath);
	~RealSenseViewer();

	void run();

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

	int * temporaryDepthValues_;
	int temporaryDepthValuesSize_;

};


#endif // !REALSENSE_VIEWER
