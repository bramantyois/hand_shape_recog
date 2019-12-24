#ifndef K_MEANS_SEGMENTATION
#define K_MEANS_SEGMENTATION
#include <opencv/cv.hpp>
#include "SkinColorBasedSegmentation.h"

cv::Mat getKmeansSegmentation(cv::Mat * sourceImage, const int numOfClusters, const int maxIter, const int maxError, const int numOfAttempts)
{
	cv::Mat src = sourceImage->clone();

	cv::Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*src.rows, z) = src.at<cv::Vec3b>(y, x)[z];
	
	cv::Mat labels;
	cv::Mat centers;
	kmeans(samples, numOfClusters, labels, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, maxIter, maxError), numOfAttempts, cv::KMEANS_PP_CENTERS, centers);
	
	cv::Mat kmeansResultImage(src.size(), src.type());
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x*src.rows, 0);
			kmeansResultImage.at <cv::Vec3b > (y, x)[0] = centers.at<float>(cluster_idx, 0);
			kmeansResultImage.at<cv::Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			kmeansResultImage.at<cv::Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	}
	//imshow("kmeans segmentation", kmeansResultImage);
	//cv::GaussianBlur(kmeansResultImage, kmeansResultImage, cv::Size(7, 7), 1, 1);
	//cv::Mat mask= getSkinMaskHSV(&kmeansResultImage);
	cv::Mat mask;
	cv::cvtColor(kmeansResultImage, mask, CV_RGB2GRAY);
	cv::threshold(mask, mask, 100, 255, CV_THRESH_BINARY);
	//erosion(mask, mask, 3);
	dilation(mask, mask,3);

	/*cv::Mat invMask;
	cv::bitwise_not(mask, invMask);
	cv::Mat floodfill = invMask.clone();
	cv::floodFill(invMask, cv::Point(0, 0), cv::Scalar(255));
	cv::Mat invfloodfill;
	cv::bitwise_not(floodfill, invfloodfill);
	cv::Mat result = (mask | invfloodfill);

	mask = result.clone();*/
	cv::imshow("mask", mask);

	cv::Mat input = sourceImage->clone();

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


cv::Mat getKmeansSegmentationHSV(cv::Mat * sourceImage, const int numOfClusters, const int maxIter, const int maxError, const int numOfAttempts)
{
	cv::Mat rgbSrc = sourceImage->clone();
	//to HSV Space
	cv::Mat hsvSrc;
	cvtColor(rgbSrc, hsvSrc, CV_BGR2HSV);

	cv::Mat samples(hsvSrc.rows * hsvSrc.cols, 3, CV_32F);
	for (int y = 0; y < hsvSrc.rows; y++)
		for (int x = 0; x < hsvSrc.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*hsvSrc.rows, z) = hsvSrc.at<cv::Vec3b>(y, x)[z];

	cv::Mat labels;
	cv::Mat centers;
	kmeans(samples, numOfClusters, labels, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, maxIter, maxError), numOfAttempts, cv::KMEANS_PP_CENTERS, centers);

	cv::Mat kmeansResultImage(hsvSrc.size(), hsvSrc.type());
	for (int y = 0; y < hsvSrc.rows; y++)
	{
		for (int x = 0; x < hsvSrc.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x*hsvSrc.rows, 0);
			kmeansResultImage.at<cv::Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			kmeansResultImage.at<cv::Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			kmeansResultImage.at<cv::Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	}
	imshow("kmeans segmentation HSV", kmeansResultImage);
	cv::Mat maskImage;
	maskImage = getSkinMaskHSV(&kmeansResultImage);
	return maskImage;
}
#endif