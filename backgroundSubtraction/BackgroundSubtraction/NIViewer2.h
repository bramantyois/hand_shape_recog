#pragma once

#ifndef NI_VIEWER2
#define NI_VIEWER2

#include <OpenNI.h>
#include <opencv2/opencv.hpp>

class NIViewer2
{
public:
	NIViewer2(const char* strSampleName, openni::Device& device, openni::VideoStream& depthStream, openni::VideoStream& colorStream);
	virtual ~NIViewer2();

	virtual openni::Status init();
	void run();

	cv::Mat getColorMat() { return colorFrame_; }
	cv::Mat getDepthMat() { return depthFrame_; }
	cv::Mat getDepthMat8() { return depthFrame8_; }

private:
	NIViewer2(const NIViewer2&);
	NIViewer2& operator=(NIViewer2&);

	static NIViewer2* self_;

	int			frameWidth_;
	int			frameHeight_;

	cv::Mat colorFrame_;
	cv::Mat depthFrame_;
	cv::Mat depthFrame8_;

	openni::VideoFrameRef		depthFrameRef_;
	openni::VideoFrameRef		colorFrameRef_;

	openni::Device&	device_;
	openni::VideoStream& depthStream_;
	openni::VideoStream&  colorStream_;
	openni::VideoStream** videoStream_;

};


#endif