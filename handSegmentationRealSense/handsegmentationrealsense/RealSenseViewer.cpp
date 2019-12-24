#include "RealSenseViewer.h"

RealSenseViewer::RealSenseViewer(std::string folderPath):
	folderPath_(folderPath),
	tempDepthValues_(NULL),
	initilized_(false)
{
	indexAll_ = 0;
	indexDepth_ = 0;
	indexRGB_ = 0;
	
	//enumerate files, putting files to array
	cv::glob(folderPath_, filenames_);

	for (size_t i = 0; i < filenames_.size(); i++)
	{
		if (filenames_[i].find("dep") != -1)
			filenamesDepth_.push_back(filenames_[i]);
		else
			filenamesRGB_.push_back(filenames_[i]);
	}

	//initialize some arrays
	{
		cv::Mat tempDepth = cv::imread(filenamesDepth_[0]);
		if (tempDepth.data)
		{
			frameHeight_ = tempDepth.size().height;
			frameWidth_ = tempDepth.size().width;

			tempDepthValuesSize_ = frameHeight_*frameWidth_;

			tempDepthValues_ = new int[tempDepthValuesSize_];
			memset(tempDepthValues_, 0, tempDepthValuesSize_ * sizeof(int));

			colorFrame_ = cv::Mat::zeros(frameHeight_, frameWidth_, CV_8UC3);
			depthFrame_ = cv::Mat::zeros(frameHeight_, frameWidth_, CV_32SC1);
			depthFrame8_ = cv::Mat::zeros(frameHeight_, frameWidth_, CV_8UC1);

			initilized_ = true;
		}
	}
}

RealSenseViewer::~RealSenseViewer()
{
	if (tempDepthValues_) delete[] tempDepthValues_;
}

bool RealSenseViewer::run()
{
	bool depthFrameFounded = false;
	bool colorFrameFounded = false;
	if (indexDepth_ < filenamesDepth_.size() && initilized_)
	{
		cv::Mat tempDepthFrame = cv::imread(filenamesDepth_[indexDepth_]);
	//	if (tempDepthFrame.type() != CV_)std::printf("%i",tempDepthFrame.type());
		if (tempDepthFrame.data)
		{
			calculateHistogram(depthHistogram_, MAX_DEPTH, tempDepthFrame);
			
			memset(tempDepthValues_, 0, tempDepthValuesSize_* sizeof(int));
			int tempDepthIndex = 0;
			
			const uint16_t* depthPointer = (const uint16_t*)tempDepthFrame.data;
			for (int y = 0; y < frameHeight_; ++y)
			{
				for (int x = 0; x < frameWidth_; ++x, ++depthPointer, ++tempDepthIndex/*, ++texturePointer*/)
				{
					if (*depthPointer != 0)
					{
						int nHistValue = depthHistogram_[*depthPointer];
						tempDepthValues_[tempDepthIndex] = nHistValue;
					}
				}
			}
			memcpy(depthFrame_.data, tempDepthValues_, tempDepthValuesSize_ * sizeof(int));
			depthFrame_.convertTo(depthFrame8_, CV_8UC1, 1.f);
			depthFrameFounded = true;
			indexDepth_++;
		}

		//tempDepthFrame.convertTo(depthFrame8_, CV_8UC1);

		//cv::Mat equalizedFrame;
		//cv::cvtColor(tempDepthFrame, tempDepthFrame, CV_BGR2GRAY);
		//cv::equalizeHist(tempDepthFrame, equalizedFrame);
		//equalizedFrame.convertTo(depthFrame8_, CV_8UC1, 1.f);
		//depthFrameFounded = true;
		//indexDepth_++;	
	}

	if (indexRGB_ < filenamesRGB_.size())
	{
		colorFrame_ = cv::imread(filenamesRGB_[indexRGB_]);
		indexRGB_++;

		if (colorFrame_.data)
			colorFrameFounded = true;
	}
	return depthFrameFounded && colorFrameFounded;
}


