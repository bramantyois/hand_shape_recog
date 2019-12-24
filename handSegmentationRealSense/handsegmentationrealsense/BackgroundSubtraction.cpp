#include "BackgroundSubtraction.h"

BackgroundSubtraction::BackgroundSubtraction()
{
	mog2_ = cv::createBackgroundSubtractorMOG2();
}

void BackgroundSubtraction::processFrame(cv::Mat* inputFrame)
{
	inputFrame_ = inputFrame->clone();
	mog2_->apply(inputFrame_, maskFrame_);

	cv::Mat sepFrames[3];

	cv::split(inputFrame_, sepFrames);

	cv::bitwise_and(sepFrames[0], maskFrame_, sepFrames[0]);
	cv::bitwise_and(sepFrames[1], maskFrame_, sepFrames[1]);
	cv::bitwise_and(sepFrames[2], maskFrame_, sepFrames[2]);

	std::vector <cv::Mat> arrayToMerge;
	arrayToMerge.push_back(sepFrames[0]);
	arrayToMerge.push_back(sepFrames[1]);
	arrayToMerge.push_back(sepFrames[2]);

	cv::merge(arrayToMerge, segmentedFrame_);
}

