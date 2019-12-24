#include <Windows.h>
#include <direct.h>
#include <stdlib.h>  

#include "OpenNIutilities.h"
#include "NIViewer.h"
#include "preProcessing.h"
#include "Parameters.h"

int initOpenNICamera(
	int argc, char** argv,
	openni::Device * device, 
	openni::VideoStream* depthStream, 
	openni::VideoStream *colorStream)
{	
	const char* deviceURI = openni::ANY_DEVICE;
	if (argc > 1)	deviceURI = argv[1];

	openni::Status status = openni::OpenNI::initialize();
	printf("initializing...\r\n");
	status = openni::OpenNI::initialize();
	if (status != openni::STATUS_OK)
	{
		printf("init failed!\r\n");
		return 1;
	}

	printf("opening device_...\r\n");
	status = device->open(deviceURI);
	if (status != openni::STATUS_OK)
	{
		printf("device_ cannot be opened!\r\n");
		return 1;
	}

	//Assigning Devices
	printf("assigning device_ to depth stream\r\n");
	status = depthStream->create(*device, openni::SENSOR_DEPTH);
	if (status != openni::STATUS_OK)
	{
		printf("depthStream failed!\r\n");
		return 1;
	}

	printf("assigning device_ to color stream\r\n");
	status = colorStream->create(*device, openni::SENSOR_COLOR);
	if (status != openni::STATUS_OK)
	{
		printf("colorStream failed!\r\n");
		return 1;
	}

	//Setting image registration mode
	device->setImageRegistrationMode(openni::ImageRegistrationMode::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

	//starting stream and plug streams into niviewer
	status = depthStream->start();
	if (status != openni::STATUS_OK)
	{
		printf("depth stream can be started!\r\n");
		return 1;
	}

	status = colorStream->start();
	if (status != openni::STATUS_OK)
	{
		printf("color stream can be started!\r\n");
		return 1;
	}

	return 0;
}

int initOpenNIOniFile(
	int argc, char** argv,
	openni::Device *device,
	openni::VideoStream* depthStream,
	openni::VideoStream *colorStream)
{		
	openni::Status status = openni::OpenNI::initialize();
	if (status != openni::STATUS_OK) return 1;

	std::string filename;
	printf("put in .oni filename (put the extension too!): \r\n ");
	std::getline(std::cin, filename);

	std::string deviceAddress = filename;
	status = device->open(deviceAddress.c_str());
	if (status != openni::STATUS_OK)	return 1;

	//when using recorded file, uncomment this
	status = device->getPlaybackControl()->setSpeed(-1);
	if(status != openni::STATUS_OK) return 1;

	status = colorStream->create(*device, openni::SENSOR_COLOR);
	if (status != openni::STATUS_OK) return 1;

	status = depthStream->create(*device, openni::SENSOR_DEPTH);
	if (status != openni::STATUS_OK) return 1;
	
	colorStream->start();
	depthStream->start();

	return 0;
}

int main(int argc, char** argv)
{
	openni::Device device_;
	openni::VideoStream depthStream_, colorStream_;
	openni::VideoFrameRef depthFrameRead_, colorFrameRead_;
	openni::Status status_ = openni::STATUS_OK;

	cv::Mat * colorFrame_ = NULL;
	cv::Mat * depthFrame_ = NULL;

	initOpenNIOniFile(argc, argv, &device_, &depthStream_, &colorStream_);
	//initOpenNICamera(argc, argv, &device_, &depthStream_, &colorStream_);
	
	NIViewer matsContainer_("MATs Container", device_, depthStream_, colorStream_);
	status_ = matsContainer_.init();
	if (status_ != openni::STATUS_OK)
	{
		printf("mats container failed to be initialized!\r\n");
		return 0;
	}
	//////////////////////
	// VIDEO PROCESSING //
	//////////////////////
	
	std::string filename;
	printf("put in filename: \r\n ");
	std::getline(std::cin, filename);

	//creating directory
	std::string dirName = ".//";
	dirName.append(filename);
	int dilenamsuccess = _mkdir(dirName.c_str());
	
	std::string filenameColor = filename;
	filenameColor.append("_color_");
	std::string filenameDepth = filename;
	filenameDepth.append("_depth_");

	int fileIndex = 0;

	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(9);

	Preprocessing preprocessing(cv::Size(320, 240), cv::Size(21,21),7,7,3.75,6);

	while (1)
	{
		matsContainer_.run();

		cv::Mat colorMat = matsContainer_.getColorMat();
		cv::Mat depthMat = matsContainer_.getDepthMat8();
		cv::Mat eqedDepthMat = matsContainer_.getEqedeDepth8();
				
		/*bool preprocessingSuccess = preprocessing.preProcess(&colorMat, &depthMat);
		cv::Mat croppedColorL = preprocessing.getCroppedColorLeft(); 
		cv::Mat croppedColorR = preprocessing.getCroppedColorRight();
		cv::Mat croppedDepthL = preprocessing.getCroppedDepthLeft();
		cv::Mat croppedDepthR = preprocessing.getCroppedDepthRight();*/

		bool preprocessingSuccess = preprocessing.preProcessEqed(&colorMat, &depthMat, &eqedDepthMat);
		cv::Mat croppedColorL = preprocessing.getCroppedColorLeft();
		cv::Mat croppedColorR = preprocessing.getCroppedColorRight();
		cv::Mat croppedDepthL = preprocessing.getCroppedDepthLeft();
		cv::Mat croppedDepthR = preprocessing.getCroppedDepthRight();
		//Color Processing 
		if (SHOW_COLOR_IMAGE) {
			cv::imshow("croppedColorL", croppedColorL);
			cv::imshow("croppedColorR", croppedColorR);
			cv::imshow("croppedDepthL", croppedDepthL);
			cv::imshow("croppedDepthR", croppedDepthR);
			cv::imshow("InputImage", colorMat);
		}

		//Depth Processing
		if (SHOW_DEPTH_IMAGE) {
			//cv::imshow("croppedDepth", croppedDepth);
			//cv::imshow("depthMatEq", depthMatEq);
		}

		fileIndex++;
		int numberofZeropadding = 3;

		if (fileIndex > 9 && fileIndex <= 99)
			numberofZeropadding = 2;
		else if (fileIndex > 99 && fileIndex <= 999)
			numberofZeropadding = 1;
		else if (fileIndex > 999)
			numberofZeropadding = 0;

		if (numberofZeropadding >= 0)
		{
			//separately write images
			if (WRITE_COLOR_IMAGES)
			{
				std::string colorFilename = dirName;
				colorFilename.append("/").append(filenameColor).append(numberofZeropadding, '0').append(std::to_string(fileIndex)).append(".jpg");
				printf("%s\n",colorFilename.c_str());
				cv::imwrite(colorFilename, croppedColorL);
			}

			if (WRITE_DEPTH_IMAGES)
			{
				std::string depthFilename = dirName;
				depthFilename.append("/").append(filenameDepth).append(numberofZeropadding, '0').append(std::to_string(fileIndex)).append(".jpg");
				printf("%s\n",depthFilename.c_str());
				cv::imwrite(depthFilename, croppedDepthL);
			}
		}
		
		if (cvWaitKey(30) >= 0) break;
	}
	
	//outputVideo.release();
	depthStream_.destroy();
	colorStream_.destroy();
	device_.close();
	openni::OpenNI::shutdown();
	return 0;
}