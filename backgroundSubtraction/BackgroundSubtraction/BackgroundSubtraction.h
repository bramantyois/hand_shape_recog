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

class SimpleSubtractor
{
public:
	SimpleSubtractor(int numOfIntegration, int threshold);
	//~SimpleSubtractor();
	
	void processFrame(cv::Mat * frame);
	bool isBackgroundValid() {return isBackgroundValid_;};

	cv::Mat getMask() { return maskFrame_; };
	cv::Mat getSegmented() { return segmentedFrame_; };

private:
	cv::Size backgroundSize_;
	int backgroundType_;

	cv::Mat backgroundFrame_;
	cv::Mat backgroundBuffer_;
	cv::Mat backgroundFrameBW_;

	cv::Mat segmentedFrame_;
	cv::Mat maskFrame_;
	cv::Mat maskFrameBW_;

	bool isBackgroundValid_;
	bool isFirstTime_;
	int counter_;

	bool skipOK_;
	int skipFrameCounter_;

	int numOfIntegration_;
	float alpha_;

	int threshold_;
};


#endif // !BACKGROUNDSUBTRACTOR
