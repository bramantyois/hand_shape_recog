#include "RealSenseViewer.h"
#include "OniSampleUtilities.h"


RealSenseViewer::RealSenseViewer(std::string folderPath)
:folderPath_(folderPath)
{
	indexAll_ = 0;
	indexDepth_ = 0;
	indexRGB_ = 0;
	
	//enumerate files, putting files to array
	cv::glob(folderPath_, filenames_);

	for (size_t i = 0; i < filenames_.size(); i++)
	{
		std::string headname = filenames_[i].substr(0, 3);
		if (headname == "dep")
			filenamesDepth_.push_back(filenames_[i]);
		else
			filenamesRGB_.push_back(filenames_[i]);
	}
}

RealSenseViewer::~RealSenseViewer()
{

}

void RealSenseViewer::run()
{
	if (indexDepth_ < filenamesDepth_.size())
	{
		depthFrame_ = cv::imread(filenamesDepth_[indexDepth_]);
		indexDepth_++;
		if (depthFrame_.data)
			depthFrame_.convertTo(depthFrame8_, CV_8UC1, 1.f);
	}

	if (indexRGB_ < filenamesRGB_.size())
	{
		colorFrame_ = cv::imread(filenamesRGB_[indexRGB_]);
		indexRGB_++;
	}
}


