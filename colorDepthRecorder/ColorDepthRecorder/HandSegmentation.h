#pragma once

#ifndef HAND_SEGMENTATION
#define HAND_SEGMENTATION

#include "MahalSkinSegmentation.h"
#include "MorphologicalOperation.h"

class HandSegmentation
{
public:
	HandSegmentation(cv::Size blurSize, int dilationRadius, int erosionRadius, float mahalDistance);
	~HandSegmentation();

	void process(cv::Mat * colormat, cv::Mat * depthmat);
	cv::Mat getCroppedColor() { return croppedColor_; };

private:
	MahalSkinSegmentation * skinSegmentation_;
	
	cv::Mat croppedColor_;

	cv::Size resultSize_;
	cv::Size blurSize_;
	
	int dilationRadius_;
	int erosionRadius_;
};

#endif