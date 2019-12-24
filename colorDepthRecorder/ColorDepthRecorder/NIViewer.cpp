#include "NIViewer.h"
#include "OniSampleUtilities.h"

#define GL_WIN_SIZE_X	1280
#define GL_WIN_SIZE_Y	1024
#define TEXTURE_SIZE	512

#define MIN_NUM_CHUNKS(data_size, chunk_size)	((((data_size)-1) / (chunk_size) + 1))
#define MIN_CHUNKS_SIZE(data_size, chunk_size)	(MIN_NUM_CHUNKS(data_size, chunk_size) * (chunk_size))

NIViewer* NIViewer::self_ = NULL;

NIViewer::NIViewer(const char* strSampleName, openni::Device& device, openni::VideoStream& depth, openni::VideoStream& color) :
	device_(device), 
	depthStream_(depth), 
	colorStream_(color), 
	videoStream_(NULL),
	temporaryDepthValues_(NULL)
{
	self_ = this;
}

NIViewer::~NIViewer()
{
	if (temporaryDepthValues_) delete[] temporaryDepthValues_;

	self_ = NULL;
	if (videoStream_) delete[]videoStream_;
}

openni::Status NIViewer::init()
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
	
	temporaryDepthValuesSize_ = frameWidth_*frameHeight_;
	temporaryDepthValues_ = new uint16_t[temporaryDepthValuesSize_];
	
	colorFrame_ = cv::Mat::zeros( frameHeight_, frameWidth_, CV_8UC3 );
	depthFrame_ = cv::Mat::zeros( frameHeight_, frameWidth_, CV_16UC1);
	depthFrameEqed_ = cv::Mat::zeros(frameHeight_, frameWidth_, CV_16UC1);
	depthFrame8_ = cv::Mat::zeros(frameHeight_, frameWidth_, CV_8UC1);
	depthFrame8Eqed_ = cv::Mat::zeros(frameHeight_, frameWidth_, CV_8UC1);

	return openni::STATUS_OK;
}

void NIViewer::run()
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
		calculateHistogram(depthHistogram_, MAX_DEPTH, depthFrameRef_);
		
		const openni::DepthPixel* depthRowPointer = (const openni::DepthPixel*)depthFrameRef_.getData();
		memcpy(depthFrame_.data, depthRowPointer, frameHeight_ * frameHeight_ * sizeof(uint16_t));

		int rowSize = depthFrameRef_.getStrideInBytes() / sizeof(openni::DepthPixel);
		memset(temporaryDepthValues_, 0, temporaryDepthValuesSize_ *sizeof(uint16_t));
		int temporaryDepthValuesIndex = 0;
		
		for (int y = 0; y < depthFrameRef_.getHeight(); ++y)
		{
			const openni::DepthPixel* depthPointer = depthRowPointer;
			
			for (int x = 0; x < depthFrameRef_.getWidth(); ++x, ++depthPointer, ++temporaryDepthValuesIndex/*, ++texturePointer*/)
			{
				if (*depthPointer != 0)
				{
					//depthFrame_.at<char>(x, y) = *depthPointer;
					int nHistValue = (int)depthHistogram_[*depthPointer];
					temporaryDepthValues_[temporaryDepthValuesIndex] = nHistValue;
				}
			}
			depthRowPointer += rowSize;
		}
			
		memcpy(depthFrameEqed_.data, temporaryDepthValues_, temporaryDepthValuesSize_ * sizeof(uint16_t));
		
		depthFrame_.convertTo(depthFrame8_, CV_8UC1, 1/128.f);
		depthFrameEqed_.convertTo(depthFrame8Eqed_, CV_8UC1, 1.f);	
	}
}

