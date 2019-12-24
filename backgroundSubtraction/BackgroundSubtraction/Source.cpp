#include "OpenNIutilities.h"
#include "NIViewer.h"
#include "Parameters.h"
#include "BackgroundSubtraction.h"
#include "ImageProcessingUtilities.h"

int main(int argc, char** argv)
{
	Device device_;
	VideoStream depthStream_, colorStream_;
	VideoFrameRef depthFrameRead_, colorFrameRead_;
	Status status_ = STATUS_OK;

	Mat * colorFrame_ = NULL;
	Mat * depthFrame_ = NULL;

	const char* deviceURI = openni::ANY_DEVICE;
	if (argc > 1)	deviceURI = argv[1];

	printf("initializing...\r\n");
	status_ = OpenNI::initialize();
	if (status_ != STATUS_OK)
	{
		printf("init failed!\r\n");
		return 1;
	}

	printf("opening device...\r\n");
	status_ = device_.open(deviceURI);
	if (status_ != STATUS_OK)
	{
		printf("device cannot be opened!\r\n");
		return 1;
	}

	//Assigning Devices
	printf("assigning device to depth stream\r\n");
	status_ = depthStream_.create(device_, openni::SENSOR_DEPTH);
	if (status_ != STATUS_OK)
	{
		printf("depthStream failed!\r\n");
		return 1;
	}

	printf("assigning device to color stream\r\n");
	status_ = colorStream_.create(device_, openni::SENSOR_COLOR);
	if (status_ != STATUS_OK)
	{
		printf("colorStream failed!\r\n");
		return 1;
	}

	//Setting image registration mode
	device_.setImageRegistrationMode(ImageRegistrationMode::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	
	printf("\r\nCurrent Depth Stream Video Mode\r\n");
	printStreamVideoMode(&depthStream_);
	printf("\r\nCurrent Color Stream Video Mode\r\n");
	printStreamVideoMode(&colorStream_);

	
	Size colorFrameSize(colorStream_.getVideoMode().getResolutionX(), colorStream_.getVideoMode().getResolutionY());
	VideoWriter colorVideoWriter(COLOR_VIDEO_NAME, -1, colorStream_.getVideoMode().getFps(), colorFrameSize, true);
	if (!colorVideoWriter.isOpened())
	{
		printf(" Could not open the output video for write: ");
		return -1;
	}
	
	Size depthFrameSize(depthStream_.getVideoMode().getResolutionX(), depthStream_.getVideoMode().getResolutionY());
	VideoWriter depthVideoWriter(DEPTH_VIDEO_NAME,-1,depthStream_.getVideoMode().getFps(), depthFrameSize,false);
	if (!depthVideoWriter.isOpened())
	{
		printf(" Could not open the output video for write: ");
		return -1;
	}
	
	//starting stream and plug streams into niviewer
	status_ = depthStream_.start();
	if (status_ != STATUS_OK)
	{
		printf("depth stream can be started!\r\n");
		return 1;
	}

	status_ = colorStream_.start();
	if (status_ != STATUS_OK)
	{
		printf("color stream can be started!\r\n");
		return 1;
	}
	NIViewer matsContainer_("MATs Container", device_, depthStream_, colorStream_);
	status_ = matsContainer_.init();

	if (status_ != STATUS_OK)
	{
		printf("mats container failed to be initialized!\r\n");
		return 0;
	}

	//Setting up auto white balance and auto exposure
	{
		CameraSettings camSetting = *(colorStream_.getCameraSettings());

		camSetting.setAutoExposureEnabled(COLOR_AUTO_EXPOSURE);			
		camSetting.setAutoWhiteBalanceEnabled(COLOR_AUTO_WHITE_BALANCE);	

		//camSetting.setExposure(EXPOSURE);
		//camSetting.setExposure(GAIN);
	
		if (camSetting.getAutoExposureEnabled())
			printf("Auto exposure ON\r\n");
		else
			printf("Auto exposure OFF\r\n");
				
		if (camSetting.getAutoWhiteBalanceEnabled())
			printf("Auto White Balance ON\r\n");
		else
			printf("Auto White Balance OFF\r\n");

		printf("Exposure : %d \n", camSetting.getExposure());
		printf("Gain: %d \n", camSetting.getGain());
	}
	
	//////////////////////
	// VIDEO PROCESSING //
	//////////////////////
	
	while (1)
	{		
		matsContainer_.run();

		Mat colorMat = matsContainer_.getColorMat() ;
		Mat depthMat = matsContainer_.getDepthMat8();
		
		//imshow("input", colorMat);

		//Color Processing 
		if (USE_COLOR_CAMERA)
		{
			imshow("color", colorMat);
		}

		//Depth Processing
		if (USE_DEPTH_CAMERA)
		{
			imshow("depth", depthMat);
		}

		if (WRITE_COLOR_VIDEO)
		{
			if (colorMat.empty()) break;
			colorVideoWriter.write(colorMat);
		}

		if (WRITE_DEPTH_VIDEO)
		{
			if (depthMat.empty()) break;
			depthVideoWriter.write(depthMat);
		}

		if (cvWaitKey(30) >= 0)
		{
			colorVideoWriter.release();
			break;
		}
	}

	 //processing images
	if (WRITE_COLOR_IMAGES)
	{
		cv::VideoCapture capToImage(COLOR_VIDEO_NAME);
		SimpleSubtractor bgsub(60, 6);
		cv::Size outputSize(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT);

		int fileIndex = 0;
		for (;;)
		{
			cv::Mat frame;
			capToImage >> frame;
			if (frame.empty()) break;

			//segmenting image
			 bgsub.processFrame(&frame);
		
			 if (bgsub.isBackgroundValid())
			 {
				 cv::Mat segmented = bgsub.getSegmented();

				 //merge(arrayToMerge, frame);
				 cv::Mat result;
				 if (segmented.size() != outputSize)
				 {
					 cv::Mat tempMat;
					 cv::resize(segmented, tempMat, outputSize);

					 cv::flip(tempMat, result, 1);
				 }
				 else
				 {
					 cv::flip(segmented, result, 1);
				 }

				 simpleColorBalancing(result, result, 1);

				 cv::imshow("segmented image", result);

				 //naming and writing image
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
					 std::string fileName = std::string("image").append(numberofZeropadding, '0').append(std::to_string(fileIndex)).append(".png");
					 std::printf(fileName.c_str());
					 std::vector<int> compression_params;
					 compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
					 compression_params.push_back(9);
					 cv::imwrite(fileName.c_str(), result, compression_params);
				 }
			 }			
		}
	}

	if (WRITE_DEPTH_IMAGES)
	{

	}


	//outputVideo.release();
	depthStream_.destroy();
	colorStream_.destroy();
	device_.close();
	OpenNI::shutdown();
	return 0;
}