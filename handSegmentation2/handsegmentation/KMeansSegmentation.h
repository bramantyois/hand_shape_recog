#ifndef K_MEANS_SEGMENTATION
#define K_MEANS_SEGMENTATION
#include <opencv/cv.hpp>
#include "SkinColorBasedSegmentation.h"
using namespace cv;
using namespace std;

Mat getKmeansSegmentation(Mat * sourceImage, const int numOfClusters, const int maxIter, const int maxError, const int numOfAttempts)
{
	Mat src = sourceImage->clone();

	Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y, x)[z];
	
	Mat labels;
	Mat centers;
	kmeans(samples, numOfClusters, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, maxIter, maxError), numOfAttempts, KMEANS_PP_CENTERS, centers);
	
	Mat kmeansResultImage(src.size(), src.type());
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x*src.rows, 0);
			kmeansResultImage.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			kmeansResultImage.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			kmeansResultImage.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	}
	imshow("kmeans segmentation", kmeansResultImage);
	GaussianBlur(kmeansResultImage, kmeansResultImage, Size(7, 7), 1, 1);
	Mat maskImage;
	maskImage = getSkinMaskHSV(&kmeansResultImage);
	return maskImage;
}


Mat getKmeansSegmentationHSV(Mat * sourceImage, const int numOfClusters, const int maxIter, const int maxError, const int numOfAttempts)
{
	Mat rgbSrc = sourceImage->clone();
	//to HSV Space
	Mat hsvSrc;
	cvtColor(rgbSrc, hsvSrc, CV_BGR2HSV);

	Mat samples(hsvSrc.rows * hsvSrc.cols, 3, CV_32F);
	for (int y = 0; y < hsvSrc.rows; y++)
		for (int x = 0; x < hsvSrc.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*hsvSrc.rows, z) = hsvSrc.at<Vec3b>(y, x)[z];

	Mat labels;
	Mat centers;
	kmeans(samples, numOfClusters, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, maxIter, maxError), numOfAttempts, KMEANS_PP_CENTERS, centers);

	Mat kmeansResultImage(hsvSrc.size(), hsvSrc.type());
	for (int y = 0; y < hsvSrc.rows; y++)
	{
		for (int x = 0; x < hsvSrc.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x*hsvSrc.rows, 0);
			kmeansResultImage.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			kmeansResultImage.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			kmeansResultImage.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	}
	imshow("kmeans segmentation HSV", kmeansResultImage);
	Mat maskImage;
	maskImage = getSkinMaskHSV(&kmeansResultImage);
	return maskImage;
}
#endif