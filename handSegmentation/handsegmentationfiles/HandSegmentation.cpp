#include <opencv2/opencv.hpp>
//#include "SkinColorBasedSegmentation.h"
//#include "KMeansSegmentation.h"
//#include "MeanShiftSegmentation.h"
#include "NIViewer.h"
#include <iostream>
#include <OpenNI.h>
#include "MahalSkinSegmentation.h"
#include "MorphologicalOperation.h"
#include "MinAreaRect.h"

#include "ParametersSetting.h"


using namespace cv;
using namespace std;	

int main(int argc, char** argv)
{
	//Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	openni::Status status = openni::OpenNI::initialize();
	if (status != openni::STATUS_OK) return 1;

	openni::Device device;
	//char* oniAddress= ".//..//testOni//record.oni";
	status = device.open(openni::ANY_DEVICE);
	//string deviceAddress=".//..//Resources//testOni//record.oni";
	//status = device.open(deviceAddress.c_str());
	if (status != openni::STATUS_OK)	return 1;

	//when using recorder file, uncomment this
	//status = device.getPlaybackControl()->setSpeed(-1);
	//if(status != openni::STATUS_OK) return 1;

	if (!(device.hasSensor(openni::SENSOR_DEPTH) && device.hasSensor(openni::SENSOR_COLOR)))return 1;
	if (device.setImageRegistrationMode(openni::ImageRegistrationMode::IMAGE_REGISTRATION_DEPTH_TO_COLOR) != openni::STATUS_OK) return 1;
	openni::VideoStream colorStream, depthStream;

	status = colorStream.create(device, openni::SENSOR_COLOR);
	if (status != openni::STATUS_OK) return 1;

	status = depthStream.create(device, openni::SENSOR_DEPTH);
	if (status != openni::STATUS_OK) return 1;

	openni::VideoMode videoMode = colorStream.getVideoMode();
	int fps = videoMode.getFps();
	int frameHeight = videoMode.getResolutionY();
	int frameWidth = videoMode.getResolutionX();
	
	//VideoWriter segementationResult("resultColor.avi", colorVideo.get(CV_CAP_PROP_FOURCC), colorVideo.get(CV_CAP_PROP_FPS), colorVideoSize, true);
	VideoWriter resultVideoWriter("result.avi", CV_FOURCC('D', 'I', 'V', 'X'), fps, Size(frameWidth,frameHeight), true);
	VideoWriter depthVideosWriter("depth.avi", CV_FOURCC('D', 'I', 'V', 'X'), fps, Size(frameWidth, frameHeight), false);
	VideoWriter maskVideoWriter("colorMask.avi", CV_FOURCC('D','I','V','X'), fps, Size(frameWidth, frameHeight), false);	
	
	MahalSkinSegmentation mahalSegmentation;
	if (!mahalSegmentation.fetchData(".//..//Resources//Skin_NonSkin.txt", true)) return 1;
	if (!mahalSegmentation.calcVarianceAndMean()) return 1;

	colorStream.start();
	depthStream.start();

	NIViewer viewer("viewer", device, depthStream, colorStream);
	status = viewer.init();
	if (status != openni::STATUS_OK) return 1;
	
	/*VideoCapture colorVideo(".//..//testvideo//color.avi");		
	if (!colorVideo.isOpened())
	{
		printf("No files");
		return 0;
	}*/

	//Size colorVideoSize(colorVideo.get(CV_CAP_PROP_FRAME_WIDTH), colorVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
	//VideoWriter segementationResult("resultColor.avi", colorVideo.get(CV_CAP_PROP_FOURCC), colorVideo.get(CV_CAP_PROP_FPS), colorVideoSize, true);
	
	//Mat colorImage;
	//while (colorVideo.read(colorImage))
	//{
	//	if (colorImage.empty())	break;
	//	
	//	Mat maskImage = getSkinMaskHSV(&colorImage);
	//	//Mat maskImage = getKmeansSegmentation(&colorImage, 4, 10, 0.01f, 5);
	//	//Mat segmentedImage = getMeanShiftedSegmentation(&image, 10, 1);
	//	//imshow("maskImage", maskImage);

	//	Mat sepImg[3], segmentedImg[3];
	//	split(colorImage, sepImg);

	//	bitwise_and(sepImg[0], maskImage, segmentedImg[0]);
	//	bitwise_and(sepImg[1], maskImage, segmentedImg[1]);
	//	bitwise_and(sepImg[2], maskImage, segmentedImg[2]);

	//	vector<Mat> arrayToMerge;
	//	arrayToMerge.push_back(segmentedImg[0]);
	//	arrayToMerge.push_back(segmentedImg[1]);
	//	arrayToMerge.push_back(segmentedImg[2]);

	//	Mat resImage;
	//	merge(arrayToMerge, resImage);
	//	segementationResult.write(resImage);
	//	//imshow("Result", resImage);
	//}

	Mat colorFrameSource;
	Mat depthFrameSource;

	Mat maskColor;
	Mat maskDepth;

	int fileIndex = 0;
	while (true)
	{
		Mat colorFrameSource;
		Mat depthFrameSource;

		Mat maskColor;
		Mat maskDepth;
		Mat finalMask;

		Mat minRectResult;

		viewer.run();
		colorFrameSource = viewer.getColorMat();
		depthFrameSource = viewer.getDepthMat8();

		imshow("colorFrame", colorFrameSource);
		imshow("depthFrame", depthFrameSource);

		//
		//{	//get mask from color
		//	maskColor = mahalSegmentation.getMahalanobisDistImg(colorFrameSource);
		//	cv::threshold(maskColor, maskColor, MAHAL_DISTANCE, 255, CV_THRESH_BINARY_INV);
		//	maskColor.convertTo(maskColor, CV_8U);

		//	//dilation(maskColor, maskColor, 4);
		//	//erosion(maskColor, maskColor, 2);
		//	imshow("maskColor", maskColor);
		//}
		//
		//{	//get mask from depth
		//	cv::threshold(depthFrameSource, maskDepth, 150, 255, CV_THRESH_BINARY);
		//	//dilation(maskDepth, maskDepth, 4);
		//	//erosion(maskDepth, maskDepth, 1);
		//	imshow("maskDepth", maskDepth);
		//}

		//{	//get mask from both color and depth
		//	bitwise_and(maskDepth, maskColor, finalMask);
		//	dilation(finalMask, finalMask, DILATION_RADIUS);
		//	erosion(finalMask, finalMask, EROSION_RADIUS);
		//	imshow("finalMask", finalMask);
		//	/*vector<int> compression_params;
		//	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		//	compression_params.push_back(9);

		//	cv::imwrite("maskDepth.png", maskDepth, compression_params);
		//	cv::imwrite("maskColor.png", maskColor, compression_params);
		//	cv::imwrite("finalMask.png", finalMask, compression_params);*/
		//}

		////get min area rectangle
		//	
		//cv::Mat subImage = getCroppedMaskedHandArea(colorFrameSource, finalMask);
		//cv::Mat croppedImage = resizeAndFitCroppedImage(subImage,cv::Size(RESULT_IMAGE_WIDTH, RESULT_IMAGE_HEIGHT));

		//imshow("drawn mask", croppedImage);		
		//
		//fileIndex++;
		//int numberofZeropadding = 3;

		//if (fileIndex > 9)
		//	numberofZeropadding = 2;
		//else if (fileIndex > 99)
		//	numberofZeropadding = 1;
		//else if (fileIndex > 999)
		//	numberofZeropadding = 0;
		//
		//if (numberofZeropadding >= 0)
		//{
		//	std::string fileName = std::string("bram").append(numberofZeropadding,'0').append(std::to_string(fileIndex)).append(".png");
		//	std::printf(fileName.c_str());
		//	vector<int> compression_params;
		//	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		//	compression_params.push_back(9);
		//	imwrite(fileName.c_str(), croppedImage, compression_params);
		//}


		/*Mat sepImg[3], segmentedImg[3];
		split(colorFrameSource, sepImg);

		bitwise_and(sepImg[0], maskColor, segmentedImg[0]);
		bitwise_and(sepImg[1], maskColor, segmentedImg[1]);
		bitwise_and(sepImg[2], maskColor, segmentedImg[2]);

		bitwise_and(segmentedImg[0], depthFrameSource, segmentedImg[0]);
		bitwise_and(segmentedImg[1], depthFrameSource, segmentedImg[1]);
		bitwise_and(segmentedImg[2], depthFrameSource, segmentedImg[2]);

		vector<Mat> arrayToMerge;
		arrayToMerge.push_back(segmentedImg[0]);
		arrayToMerge.push_back(segmentedImg[1]);
		arrayToMerge.push_back(segmentedImg[2]);

		Mat segmentedImage;
		merge(arrayToMerge, segmentedImage);

		imshow("result", segmentedImage);
		imshow("input image", colorFrameSource);
		imshow("mask color", maskColor);
		imshow("mask depth", depthFrameSource);
		imshow("depth", colorFrameSource);

		resultVideoWriter.write(segmentedImage);
		maskVideoWriter.write(maskColor);
		depthVideosWriter.write(depthFrameSource);*/
		if (cvWaitKey(30) >= 0)
		{
			break;
		}
	}

	waitKey();
	device.close();
	openni::OpenNI::shutdown();
	return 0;
}