#include "BackgroundSubtraction.h"
#include "MorphologicalOperation.h"

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


SimpleSubtractor::SimpleSubtractor(int numOfIntegration, int threshold)
{
	isBackgroundValid_ = false;
	isFirstTime_ = true;
	numOfIntegration_ = numOfIntegration;
	alpha_ = 1.f / numOfIntegration_;
	counter_ = 0;

	threshold_ = threshold;

	skipFrameCounter_ = 0; 
	skipOK_ = false;
}

void SimpleSubtractor::processFrame(cv::Mat * inputFrame)
{
	if (!skipOK_)
	{
		segmentedFrame_ = inputFrame->clone();
		maskFrame_ = inputFrame->clone();
		cv::cvtColor(maskFrame_,maskFrameBW_, CV_8UC1);
		if (++skipFrameCounter_ > 10)
			skipOK_ = true;
	}

	if (inputFrame->data && skipOK_)
	{		
		if (isFirstTime_)
		{//initialize background MAT
			backgroundSize_ = inputFrame->size();
			backgroundType_ = inputFrame->type();
			backgroundFrame_ = cv::Mat::zeros(backgroundSize_, backgroundType_);
			backgroundBuffer_ = cv::Mat::zeros(backgroundSize_, CV_64FC3);
			cv::cvtColor(backgroundFrame_, backgroundFrameBW_, CV_BGR2GRAY);

			isFirstTime_ = false;
		}
		
		if (isBackgroundValid_)
		{
			cv::Mat input = inputFrame->clone();
			{//processing color

				cv::Mat maskColor;
				cv::absdiff(input, backgroundFrame_, maskColor);

				cv::Mat sepMask[3];
				cv::split(maskColor, sepMask);

				cv::threshold(sepMask[0], sepMask[0], threshold_, 255, CV_THRESH_BINARY);
				cv::threshold(sepMask[1], sepMask[1], threshold_, 255, CV_THRESH_BINARY);
				cv::threshold(sepMask[2], sepMask[2], threshold_, 255, CV_THRESH_BINARY);

				cv::bitwise_or(sepMask[0], sepMask[1], maskFrame_);
				cv::bitwise_or(maskFrame_, sepMask[2], maskFrame_);

				dilation(maskFrame_, maskFrame_, 3);
				erosion(maskFrame_, maskFrame_, 3);

				cv::imshow("mask", maskFrame_);

				cv::Mat sepInput[3];
				cv::split(input, sepInput);

				cv::bitwise_and(sepInput[0], maskFrame_, sepInput[0]);
				cv::bitwise_and(sepInput[1], maskFrame_, sepInput[1]);
				cv::bitwise_and(sepInput[2], maskFrame_, sepInput[2]);

				std::vector <cv::Mat> arrayToMerge;
				arrayToMerge.push_back(sepInput[0]);
				arrayToMerge.push_back(sepInput[1]);
				arrayToMerge.push_back(sepInput[2]);

				cv::merge(arrayToMerge, segmentedFrame_);
			}

			//{//processing BW
			//	cv::Mat inputBW;
			//	cv::cvtColor(input, inputBW, CV_BGR2GRAY);

			//	cv::Mat mask;
			//	cv::absdiff(inputBW, backgroundFrameBW_, mask);
			//	cv::threshold(mask, mask, threshold_, 255, CV_THRESH_BINARY);

			//	//maskFrameBW_ = mask.clone();
			//	cv::Mat sepInput[3];

			//	cv::split(input, sepInput);

			//	cv::bitwise_and(sepInput[0], mask, sepInput[0]);
			//	cv::bitwise_and(sepInput[1], mask, sepInput[1]);
			//	cv::bitwise_and(sepInput[2], mask, sepInput[2]);

			//	std::vector <cv::Mat> arrayToMerge;
			//	arrayToMerge.push_back(sepInput[0]);
			//	arrayToMerge.push_back(sepInput[1]);
			//	arrayToMerge.push_back(sepInput[2]);

			//	cv::merge(arrayToMerge, segmentedFrame_);
			//}			
		}
		else
		{
			cv::Mat input;
			inputFrame->convertTo(input, CV_64FC3);
			cv::addWeighted(backgroundBuffer_, 1.0, input, alpha_, 0, backgroundBuffer_);
			backgroundBuffer_.convertTo(backgroundFrame_, CV_8UC3);
			cv::cvtColor(backgroundFrame_, backgroundFrameBW_, CV_BGR2GRAY);
			cv::imshow("bg", backgroundFrame_);
			
			segmentedFrame_ = inputFrame->clone();
			maskFrame_ = inputFrame->clone();
			maskFrame_.convertTo(maskFrameBW_, CV_8UC1);

			counter_++;
			if (counter_ >= numOfIntegration_)
				isBackgroundValid_ = true;
		}
	}
}