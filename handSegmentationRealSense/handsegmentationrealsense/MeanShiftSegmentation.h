#ifndef MEAN_SHIFT_SEGMENTATION
#define MEAN_SHIFT_SEGMENTATION

#pragma once
#include <opencv/cv.hpp>

using namespace cv;
using namespace std;

Mat getMeanShiftedSegmentation(Mat * sourceImage, const int maxIter, const int maxError)
{
	Mat src = sourceImage->clone();

	Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y, x)[z];

	Mat labels;	
	Mat centers;
	Rect window = Rect(0, 0, 10, 10);

	meanShift(samples, window, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, maxIter, maxError));

	Mat new_image(src.size(), src.type());
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x*src.rows, 0);
			new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	return new_image;
}
#endif