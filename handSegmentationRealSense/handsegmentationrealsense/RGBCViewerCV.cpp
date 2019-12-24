#include <opencv2/opencv.hpp>
//#include "BackgroundSubtraction.h"
//#include "MahalSkinSegmentation.h"
#include "KMeansSegmentation.h"

#define FILE_DIRECTORY ".//..//Resources//RealSenseImages//"
#define VIDEO_NAME "ColorVideo.avi"

#define COLOR_WIDTH 320
#define COLOR_HEIGHT 240

#define DEPTH_WIDTH 600
#define DEPTH_HEIGHT 400

#define RECORD_MODE true

#define USE_COLOR true
#define USE_DEPTH false

#define WRITE_IMAGES false 
#define WRITE_VIDEOS false

int main(int argc, char** argv)
{
	cv::VideoCapture colorCapture;
	cv::VideoCapture depthCapture;

	cv::VideoWriter colorWriter;
	cv::VideoWriter depthWriter;

	bool colorFlag;
	{
		colorFlag = true;
		if (!colorCapture.open(0))
			colorFlag &= false;

		if (!colorCapture.set(CV_CAP_PROP_FRAME_WIDTH, COLOR_WIDTH))
			colorFlag &= false;

		if (!colorCapture.set(CV_CAP_PROP_FRAME_HEIGHT, COLOR_HEIGHT))
			colorFlag &= false;

		colorFlag &= USE_COLOR;
	}

	bool depthFlag;
	{
		depthFlag = true;

		if (!depthCapture.open(1))
			depthFlag &= false;

		if (!depthCapture.set(CV_CAP_PROP_FRAME_WIDTH, DEPTH_WIDTH))
			depthFlag &= false;

		if (!depthCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DEPTH_HEIGHT))
			depthFlag &= false;

		depthFlag &= USE_DEPTH;
	}

	bool videoOkay = true;	

	if (RECORD_MODE) 
	{
		if (colorFlag)
		{
			int fourcc = cv::VideoWriter::fourcc('D', 'I', 'V', 'X');
			int fps = 20;
			int height = colorCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
			int width = colorCapture.get(CV_CAP_PROP_FRAME_WIDTH);

			if (colorWriter.open(VIDEO_NAME, fourcc, fps, cv::Size(width, height)))
				videoOkay = true;
			else
				videoOkay = false;
		}

		for (;;)
		{
			if (colorFlag)
			{
				cv::Mat frame;
				colorCapture >> frame;
				if (frame.empty()) break; // end of video stream
			
				if (videoOkay)	colorWriter.write(frame);
				//imwrite("tet.jpg", frame);
				imshow("Color", frame);
			}

			if (depthFlag)
			{
				cv::Mat frame;
				depthCapture >> frame;
				if (frame.empty()) break; // end of video stream
				imshow("Depth", frame);
			}
			
			if (cv::waitKey(10) == 27) break; // stop capturing by pressing ESC 
		}
	}

	if (videoOkay && WRITE_IMAGES)
	{
		cv::VideoCapture capToImage(VIDEO_NAME);
		//MahalSkinSegmentation mahalSegmentation;
		//if (!mahalSegmentation.fetchData(".//..//Resources//Skin_NonSkin.txt", true)) return 1;
		//if (!mahalSegmentation.calcVarianceAndMean()) return 1;
		
		int fileIndex = 0;
		for (;;)
		{
			cv::Mat frame;
			capToImage >> frame;
			if (frame.empty()) break;

			//segmenting image
			cv::Mat segmented = getKmeansSegmentation(&frame,2,2,2,2);
			//cv::Mat thresholdMat;
			//cv::threshold(mask, thresholdMat, 4.5, 255, CV_THRESH_BINARY_INV);
			//thresholdMat.convertTo(mask, CV_8U);

			//cv::Mat sepImg[3], resImg[3];
			//cv::split(frame, sepImg);
			//cv::bitwise_and(sepImg[0], thresholdMat, resImg[0]);
			//cv::bitwise_and(sepImg[1], thresholdMat, resImg[1]);
			//cv::bitwise_and(sepImg[2], thresholdMat, resImg[2]);

			//std::vector<cv::Mat> arrayToMerge;
			//arrayToMerge.push_back(resImg[0]);
			//arrayToMerge.push_back(resImg[1]);
			//arrayToMerge.push_back(resImg[2]);

			//merge(arrayToMerge, frame);
			cv::Mat result = segmented.clone();
			cv::imshow("segmented image", result);

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

	// the camera will be closed automatically upon exit
	// cap.close();

	colorCapture.release();
	depthCapture.release();
	return 0;
}

/*

fileIndex++;
int numberofZeropadding = 3;

if (fileIndex > 9)
numberofZeropadding = 2;
else if (fileIndex > 99)
numberofZeropadding = 1;
else if (fileIndex > 999)
numberofZeropadding = 0;

if (numberofZeropadding >= 0)
{
std::string fileName = std::string("bram").append(numberofZeropadding, '0').append(std::to_string(fileIndex)).append(".png");
std::printf(fileName.c_str());
std::vector<int> compression_params;
compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
compression_params.push_back(9);
cv::imwrite(fileName.c_str(), frame, compression_params);
}
*/