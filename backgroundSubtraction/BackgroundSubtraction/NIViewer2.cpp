#pragma once
#include "NIViewer2.h"

#define MIN_DEPTH_DISTANCE 800
#define MAX_DEPTH_DISTANCE 3500

NIViewer2* NIViewer2::self_ = NULL;

NIViewer2::NIViewer2(const char* strSampleName, openni::Device& device, openni::VideoStream& depth, openni::VideoStream& color) :
	device_(device),
	depthStream_(depth),
	colorStream_(color),
	videoStream_(NULL)
{
	self_ = this;
}

NIViewer2::~NIViewer2()
{
	self_ = NULL;
	if (videoStream_) delete[]videoStream_;
}

openni::Status NIViewer2::init()
{
	openni::VideoMode depthVideoMode;
	openni::VideoMode colorVideoMode;

	if (depthStream_.isValid() && colorStream_.isValid())
	{
		depthVideoMode = depthStream_.getVideoMode();
		colorVideoMode = colorStream_.getVideoMode();

		int depthWidth = depthVideoMode.getResolutionX();
		int depthHeight = depthVideoMode.getResolutionY();
		int colorWidth = colorVideoMode.getResolutionX();
		int colorHeight = colorVideoMode.getResolutionY();

		if (depthWidth == colorWidth &&
			depthHeight == colorHeight)
		{
			frameWidth_ = depthWidth;
			frameHeight_ = depthHeight;
		}
		else
		{
			printf("Error - expect color and depth to be in same resolution: D: %dx%d, C: %dx%d\n",
				depthWidth, depthHeight,
				colorWidth, colorHeight);
			return openni::STATUS_ERROR;
		}
	}
	else if (depthStream_.isValid())
	{
		depthVideoMode = depthStream_.getVideoMode();
		frameWidth_ = depthVideoMode.getResolutionX();
		frameHeight_ = depthVideoMode.getResolutionY();
	}
	else if (colorStream_.isValid())
	{
		colorVideoMode = colorStream_.getVideoMode();
		frameWidth_ = colorVideoMode.getResolutionX();
		frameHeight_ = colorVideoMode.getResolutionY();
	}
	else
	{
		printf("Error - expects at least one of the streams to be valid...\n");
		return openni::STATUS_ERROR;
	}

	videoStream_ = new openni::VideoStream*[2];
	videoStream_[0] = &depthStream_;
	videoStream_[1] = &colorStream_;
	
	colorFrame_ = cv::Mat::zeros(frameHeight_, frameWidth_, CV_8UC3);
	depthFrame_ = cv::Mat::zeros(frameHeight_, frameWidth_, CV_16UC1);
	depthFrame8_ = cv::Mat::zeros(frameHeight_, frameWidth_, CV_8UC1);

	return openni::STATUS_OK;
}

void NIViewer2::run()
{
	int changedIndex;
	openni::Status rc = openni::OpenNI::waitForAnyStream(videoStream_, 2, &changedIndex);
	if (rc != openni::STATUS_OK)
	{
		printf("Wait failed\n");
		return;
	}

	switch (changedIndex)
	{
	case 0:
		depthStream_.readFrame(&depthFrameRef_); break;
	case 1:
		colorStream_.readFrame(&colorFrameRef_); break;
	default:
		printf("Error in wait\n");
	}

	if (colorFrameRef_.isValid())
	{
		const openni::RGB888Pixel* colorPointer = (const openni::RGB888Pixel*)colorFrameRef_.getData();
		memcpy(colorFrame_.data, colorPointer, 3 * frameHeight_ * frameWidth_ * sizeof(uint8_t));
		cv::cvtColor(colorFrame_, colorFrame_, CV_BGR2RGB);
	}

	if (depthFrameRef_.isValid())
	{
		const openni::DepthPixel* pDepth = (const openni::DepthPixel*)depthFrameRef_.getData();

		memcpy(depthFrame_.data, pDepth, frameHeight_ * frameWidth_ * sizeof(uint16_t));


		double min, max;
		cv::minMaxLoc(depthFrame_, &min, &max);

		printf("min: %d max: %d\r\n", (int)min, (int)max);
		//converting frame 
		/*{
			for (int j = 0; j < frameHeight_; j++)
			{
				for (int i = 0; i < frameWidth_; i++)
				{
					uint16_t val = depthFrame_.at<uint16_t>(j, i);
					if (val)
					{
						if (val > MAX_DEPTH_DISTANCE)
						{
							val = MAX_DEPTH_DISTANCE;
						}
						else if (val < MIN_DEPTH_DISTANCE)
						{
							val = MIN_DEPTH_DISTANCE;
						}
						 
						depthFrame_.at<uint16_t>(j, i) = MAX_DEPTH_DISTANCE - (val - MIN_DEPTH_DISTANCE);

					}
				}
			}
		}*/
		depthFrame_.convertTo(depthFrame8_,CV_8UC1,1.f/8.f);
		cv::equalizeHist(depthFrame8_, depthFrame8_);
	}
}
