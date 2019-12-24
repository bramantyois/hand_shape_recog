#pragma once
#ifndef BACKGROUNDSUBTRACTION
#define BACKGROUNDSUBTRACTION

#include <opencv2/opencv.hpp>
#include <stdio.h>

class BackgroundSubtraction
{
public:
	BackgroundSubtraction();
	//~BackgroundSubtraction();

	void processFrame(cv::Mat* frame);

	cv::Mat getMask() { return maskFrame_; };
	cv::Mat getSegmented() { return segmentedFrame_; };
	
private:
	cv::Mat inputFrame_;
	cv::Mat maskFrame_;//foreground mask mog2

	cv::Mat segmentedFrame_;


	cv::Ptr<cv::BackgroundSubtractor> mog2_;
};


#endif // !BACKGROUNDSUBTRACTOR
