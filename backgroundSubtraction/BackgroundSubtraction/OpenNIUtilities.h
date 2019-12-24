#ifndef OPENNI_UTILITIES
#define OPENNI_UTILITIES

#include <openCV2/opencv.hpp>
#include <iostream>
#include <OpenNI.h>
using namespace std;
using namespace cv;
using namespace openni;

///////////////
// Functions //
///////////////
string getPixelFormatName(int pixelFormat)
{
	string pixelFormatName;
	switch (pixelFormat)
	{
		//depth
	case PIXEL_FORMAT_DEPTH_1_MM:
		pixelFormatName.assign("Depth mode 1 MM");
		break;
	case PIXEL_FORMAT_DEPTH_100_UM:
		pixelFormatName.assign("Depth mode 100 UM");
		break;
	case PIXEL_FORMAT_SHIFT_9_2:
		pixelFormatName.assign("Depth mode Shift 9.2");
		break;
	case PIXEL_FORMAT_SHIFT_9_3:
		pixelFormatName.assign("Depth mode Shift 9.3");
		break;

		//color
	case PIXEL_FORMAT_RGB888:
		pixelFormatName.assign("Color mode RGB888");
		break;
	case PIXEL_FORMAT_YUV422:
		pixelFormatName.assign("Color mode YUV422");
		break;
	case PIXEL_FORMAT_GRAY8:
		pixelFormatName.assign("Color mode GRAY8");
		break;
	case PIXEL_FORMAT_GRAY16:
		pixelFormatName.assign("Color mode GRAY16");
		break;
	case PIXEL_FORMAT_JPEG:
		pixelFormatName.assign("Color mode JPEG");
		break;
	case PIXEL_FORMAT_YUYV:
		pixelFormatName.assign("Color mode YUYV");
		break;
	default:
		break;
	}
	return pixelFormatName;
}

void printSupportedVideoModes(VideoStream* videoStream)
{
	const SensorInfo& info = videoStream->getSensorInfo();
	const Array<VideoMode>& videoModes = info.getSupportedVideoModes();

	for (int i = 0; i < videoModes.getSize(); i++)
	{
		printf("Video Mode %i \r\n", i);
		printf("FPS    : %i \r\n", videoModes[i].getFps());
		printf("Width  : %i \r\n", videoModes[i].getResolutionX());
		printf("Height : %i \r\n", videoModes[i].getResolutionY());
		printf("Pixel Format : %s \r\n\n", getPixelFormatName(videoModes[i].getPixelFormat()).c_str());
	}
}

void printStreamVideoMode(VideoStream* videoStream)
{
	VideoMode retrievedVideoMode = videoStream->getVideoMode();
	printf("Video FPS    : %i \r\n", retrievedVideoMode.getFps());
	printf("Video Width  : %i \r\n", retrievedVideoMode.getResolutionX());
	printf("Video Height : %i \r\n", retrievedVideoMode.getResolutionY());

	printf("Pixel Format : %s \r\n\n", getPixelFormatName(retrievedVideoMode.getPixelFormat()).c_str());
}

Status setVideoMode(VideoStream* videoStream)
{
	const SensorInfo& info = videoStream->getSensorInfo();
	const Array<VideoMode>& videoModes = info.getSupportedVideoModes();
	int numberOfVideoModes = videoModes.getSize();

	if (numberOfVideoModes > 0)
	{
		for (int i = 0; i < videoModes.getSize(); i++)
		{
			printf("Video Mode %i \r\n", i);
			printf("FPS    : %i \r\n", videoModes[i].getFps());
			printf("Width  : %i \r\n", videoModes[i].getResolutionX());
			printf("Height : %i \r\n", videoModes[i].getResolutionY());
			printf("Pixel Format : %s \r\n\n", getPixelFormatName(videoModes[i].getPixelFormat()).c_str());
		}

		int choosenVideoMode = -1;
		printf("choose video mode 0-%i (integer) :\n", numberOfVideoModes - 1);
		scanf_s("%i", &choosenVideoMode);

		Status  status = videoStream->setVideoMode(videoModes[choosenVideoMode]);
		if (status != STATUS_OK) printf("setting video mode is failed!\r\n");
		return status;
	}
	else
		return STATUS_ERROR;

}

#endif // !OPENNI_UTILITIES

