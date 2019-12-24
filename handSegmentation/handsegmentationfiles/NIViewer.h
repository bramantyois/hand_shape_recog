#ifndef NI_VIEWER
#define NI_VIEWER

#include <OpenNI.h>
#include <openCV2/opencv.hpp>

#define MAX_DEPTH 10000

class NIViewer
{
public:
	NIViewer(const char* strSampleName, openni::Device& device, openni::VideoStream& depthStream, openni::VideoStream& colorStream);
	virtual ~NIViewer();

	virtual openni::Status init();
	void run();

	cv::Mat getColorMat() { return colorFrame_; }
	cv::Mat getDepthMat() { return depthFrame_; }
	cv::Mat getDepthMat8() { return depthFrame8_; }
		
private:
	NIViewer(const NIViewer&);
	NIViewer& operator=(NIViewer&);

	static NIViewer* self_;

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

	float depthHistogram_[MAX_DEPTH];

	int * temporaryDepthValues_;
	int temporaryDepthValuesSize_;
};




#endif