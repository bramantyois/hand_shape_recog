#pragma once

#ifndef MIN_AREA_RECT
#define MIN_AREA_RECT

#include <opencv2/opencv.hpp>
#include "MorphologicalOperation.h"

inline cv::RotatedRect getHandArea(cv::Mat binaryInputImage)
{
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(binaryInputImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<cv::RotatedRect> minRect(contours.size());

	int biggestRect = -1;
	float rectSizeTemp = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(cv::Mat(contours[i]));
		if (minRect[i].size.area() > rectSizeTemp)
		{
			rectSizeTemp = minRect[i].size.area();
			biggestRect = i;
		}
	}
	cv::RotatedRect resultRect = minRect[biggestRect];

	return resultRect;
}

inline cv::Mat getRotatedSubImage(cv::Mat sourceImage, cv::RotatedRect rotatedRect)
{
	float angle = rotatedRect.angle;
	cv::Size rectSize = rotatedRect.size;

	if (rotatedRect.angle < -45.f)
	{
		angle += 90;
		cv::swap(rectSize.width, rectSize.height);
	}

	cv::Mat rotationMat = getRotationMatrix2D(rotatedRect.center, angle, 1.0);
	cv::Mat rotatedSourceImage;
	warpAffine(sourceImage, rotatedSourceImage, rotationMat, sourceImage.size(), cv::INTER_CUBIC);
	cv::Mat croppedImage;
	cv::getRectSubPix(rotatedSourceImage, rectSize, rotatedRect.center, croppedImage);
	return croppedImage;
}

inline cv::Rect getPalmArea(cv::Mat binaryImage)
{
	cv::RotatedRect rotatedHandArea = getHandArea(binaryImage);
	cv::Mat subImage = getRotatedSubImage(binaryImage, rotatedHandArea);
	
	dilation(subImage, subImage, 5);

	int imageWidth = subImage.size().width;
	int imageHeight = subImage.size().height;

	bool isUpperPointFounded;
	bool isLowerPointFounded;
	int upperPoint;
	int lowerPoint;
	int tempDistance;
	int flag = 0;
	int curDistance = -1;
	for (int x = 0; x < imageWidth; x++)
	{
		isUpperPointFounded = false;
		isLowerPointFounded = false;
		upperPoint = -1;
		lowerPoint = -1;
		tempDistance = -1;
		
		for (int y = 0; y < imageHeight; y++)
		{
			if (subImage.at<int>(x,y) != 0)
			{
				isUpperPointFounded = true;
				upperPoint = y;
				break;
			}
		}

		for (int y = imageHeight - 1; y < 0; y--)
		{
			if (subImage.at<int>(x, y) != 0)
			{
				isLowerPointFounded = true;
				lowerPoint = y;
				break;
			}
		}

		if (isUpperPointFounded && isLowerPointFounded)
		{
			tempDistance = upperPoint - lowerPoint;
			if (flag == 0)
			{

			}
			else if (flag == 1)
			{

			}
			else if(flag == 2)
			{

			}
		}

	}

}

inline cv::Mat getCroppedMaskedHandArea(cv::Mat colorImage, cv::Mat binaryImage)
{	
	cv::RotatedRect rotatedHandArea= getHandArea(binaryImage);

	cv::Mat subImageColor = getRotatedSubImage(colorImage, rotatedHandArea);
	cv::Mat subImageMask = getRotatedSubImage(binaryImage, rotatedHandArea);

	cv::Mat separatedColorImg[3], segmentedColorImage[3];
	cv::split(subImageColor, separatedColorImg);
	
	cv::bitwise_and(separatedColorImg[0], subImageMask, segmentedColorImage[0]);
	cv::bitwise_and(separatedColorImg[1], subImageMask, segmentedColorImage[1]);
	cv::bitwise_and(separatedColorImg[2], subImageMask, segmentedColorImage[2]);

	std::vector<cv::Mat> arrayToMerge;
	arrayToMerge.push_back(segmentedColorImage[0]);
	arrayToMerge.push_back(segmentedColorImage[1]);
	arrayToMerge.push_back(segmentedColorImage[2]);

	cv::Mat resultImage;
	cv::merge(arrayToMerge, resultImage);

	return resultImage;
}

inline cv::Mat resizeAndFitCroppedImage(cv::Mat inputImage, cv::Size outputSize)
{	
	int inputHeight = inputImage.size().height;
	int inputWidth = inputImage.size().width;

	int outputHeight = outputSize.height;
	int outputWidth = outputSize.width;
	
	cv::Mat outputMat = cv::Mat::zeros(outputSize, inputImage.type());

	int newX = 0;
	int newY = 0;

	cv::Rect roiSrc;
	cv::Rect roiDst;

	if (inputHeight > outputHeight)
	{ //cropped image make rectangle 
		newY = (int)floor(0.5*(inputHeight - outputHeight));
		
		if (inputWidth > outputWidth)
		{//usual crop			
			roiSrc = cv::Rect( 0, newY, outputWidth, outputHeight);
			roiDst = cv::Rect(0,0, inputWidth, inputHeight);
		}
		else 
		{// inputWidth <= outputWidth
			newX = outputWidth - inputWidth ;

			roiDst = cv::Rect(newX, 0,  inputWidth, outputHeight);
			roiSrc = cv::Rect( 0, newY, inputWidth, outputHeight);
		}
	}
	else
	{
		newY = (int)floor(0.5*(outputHeight - inputHeight));

		if (inputWidth > outputWidth)
		{
			roiDst = cv::Rect( 0, newY, outputWidth,inputHeight);
			roiSrc = cv::Rect(0, 0, outputWidth, inputHeight);
		}
		else // inputWidth <= outputWidth
		{
			newX = outputWidth - inputWidth;

			roiDst = cv::Rect(newX , newY, inputWidth, inputHeight);
			roiSrc = cv::Rect(0, 0, inputWidth, inputHeight);
		}
	}

	roiDst = roiDst & cv::Rect(cv::Point(0, 0), outputMat.size());
	roiSrc = cv::Rect(roiSrc.tl(), roiDst.size());

	inputImage(roiSrc).copyTo(outputMat(roiDst));

	return outputMat;
}

inline cv::Mat getDrawnMinAreaRects(cv::Mat binaryImgInput)
{
	cv::RotatedRect resultRect = getHandArea(binaryImgInput);

	cv::Mat resultMat = binaryImgInput.clone();;
	cv::Point2f rect_points[4]; resultRect.points(rect_points);
	for (int j = 0; j < 4; j++)
		line(resultMat, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(100,100,100), 1, 8);

	return resultMat;
}

#endif // !CONVEXHULL_DETECTION
