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

cv::Mat getSkinMaskHSV(cv::Mat * imageSoure)
{
	cv::Mat3b image = imageSoure->clone();
	
	cv::cvtColor(image, image, CV_BGR2HSV);
	cv::GaussianBlur(image, image, cv::Size(7, 7), 1, 1);
	
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
	cv::threshold(imageGray, imageGray, 20, 255, CV_THRESH_BINARY);
	//erosion(imageGray, imageGray, 7); 
	//dilation(imageGray, imageGray, 7);
	return imageGray;
}


cv::Mat getSegmentedImageHSV(cv::Mat * imageSource)
{
	cv::Mat input = imageSource->clone();
	cv::Mat mask = getSkinMaskHSV(imageSource);

	cv::Mat sepFrames[3];

	cv::split(input, sepFrames);

	cv::bitwise_and(sepFrames[0], mask, sepFrames[0]);
	cv::bitwise_and(sepFrames[1], mask, sepFrames[1]);
	cv::bitwise_and(sepFrames[2], mask, sepFrames[2]);

	std::vector <cv::Mat> arrayToMerge;
	arrayToMerge.push_back(sepFrames[0]);
	arrayToMerge.push_back(sepFrames[1]);
	arrayToMerge.push_back(sepFrames[2]);

	cv::Mat segmented;
	cv::merge(arrayToMerge, segmented);
	return segmented;
}
/*
65 <Y< 170
85 <U < 140
85 < V < 160

Zaher Hamid Al-Tairi, Rahmita Wirza Rahmat, M. Iqbal Saripan, and Puteri Suhaiza Sulaiman, "Skin Segmentation Using YUV and RGB Color Spaces," Journal of Information Processing Systems, vol. 10, no. 2, pp. 283~299, 2014. DOI: 10.3745/JIPS.02.0002.
*/

cv::Mat getSkinMaskYUV(cv::Mat * imageSource)
{
	cv::Mat3b image = imageSource->clone();

	cv::cvtColor(image, image, CV_BGR2YUV);
	cv::GaussianBlur(image, image, cv::Size(7, 7), 1, 1);
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
	cvtColor(image, image, CV_YUV2BGR);
	cvtColor(image, imageGray, CV_BGR2GRAY);
	cv::threshold(imageGray, imageGray, 60, 255, CV_THRESH_BINARY);
	cv::medianBlur(imageGray, imageGray, 15);

	return imageGray;
}

#endif