#ifndef SKIN_COLOR_BASED_SEGMENTATION
#define SKIN_COLOR_BASED_SEGMENTATION

#include <opencv2\opencv.hpp>
#include "MorphologicalOperation.h"

/*

0 < H< 50
0.23 <S< 0.68
For HSV, Hue range is [0,179],
Saturation range is [0,255] and
Value range is [0,255].
*/

#define HSV_THRESHOLD 10

inline cv::Mat getSkinMaskHSV(cv::Mat * imageSoure)
{
	cv::Mat3b image = imageSoure->clone();
	
	cv::cvtColor(image, image, CV_BGR2HSV);
	cv::GaussianBlur(image, image, cv::Size(11,11), 1, 1);
	
	{
		//HSV ranges			
		for (int r = 0; r < image.rows; ++r) {
			for (int c = 0; c < image.cols; ++c)
				if ((image(r, c)[0]>1) && (image(r, c)[0] < 50) && 
					(image(r, c)[1]>58) && (image(r, c)[1]<173)); // do nothing
				else for (int i = 0; i<3; ++i)	image(r, c)[i] = 0;																									//				else for(int i=0; i<3; ++i)	frame(r,c)[i] = 0;
		}
	}

	cv::Mat1b imageGray;
	cv::cvtColor(image, image, CV_HSV2BGR);
	cv::cvtColor(image, imageGray, CV_BGR2GRAY);
	cv::threshold(imageGray, imageGray, HSV_THRESHOLD, 255, CV_THRESH_BINARY);
	return imageGray;
}

/*
65 <Y< 170
85 <U < 140
85 < V < 160

Zaher Hamid Al-Tairi, Rahmita Wirza Rahmat, M. Iqbal Saripan, and Puteri Suhaiza Sulaiman, "Skin Segmentation Using YUV and RGB Color Spaces," Journal of Information Processing Systems, vol. 10, no. 2, pp. 283~299, 2014. DOI: 10.3745/JIPS.02.0002.
*/

inline cv::Mat getSkinMaskYUV(cv::Mat * imageSource)
{
	cv::Mat3b image = imageSource->clone();

	cvtColor(image, image, CV_BGR2YUV);
	GaussianBlur(image, image, cv::Size(7, 7), 1, 1);
	//YUV ranges			
	for (int r = 0; r < image.rows; ++r) {
		for (int c = 0; c < image.cols; ++c)
			// 0<H<0.25  -   0.15<S<0.9    -    0.2<V<0.95   
			if ((image(r, c)[0]>65) && (image(r, c)[0] < 170) &&
				(image(r, c)[1]>85) && (image(r, c)[1]<140) &&
				(image(r, c)[2]>85) && (image(r, c)[2]<160)); // do nothing
			else for (int i = 0; i<3; ++i)	image(r, c)[i] = 0;																									//				else for(int i=0; i<3; ++i)	frame(r,c)[i] = 0;
	}	
	imshow("filtered yuv", image);
	cv::Mat1b imageGray;
	cv::cvtColor(image, image, CV_YUV2BGR);
	cv::cvtColor(image, imageGray, CV_BGR2GRAY);
	cv::threshold(imageGray, imageGray, 60, 255, CV_THRESH_BINARY);
	cv::medianBlur(imageGray, imageGray, 15);

	return imageGray;
}

#endif