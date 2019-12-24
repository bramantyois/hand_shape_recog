#pragma once
#include <openCV2/opencv.hpp>
#include <iostream>
/// perform the Simplest Color Balancing algorithm
 void simpleColorBalancing(cv::Mat& in, cv::Mat& out, float percent) {
	assert(in.channels() == 3);
	assert(percent > 0 && percent < 100);

	float half_percent = percent / 200.0f;

	std::vector<cv::Mat> tmpsplit; split(in, tmpsplit);
	for (int i = 0; i<3; i++) {
		//find the low and high precentile values (based on the input percentile)
		cv::Mat flat; tmpsplit[i].reshape(1, 1).copyTo(flat);
		cv::sort(flat, flat, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
		int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
		int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));
		//cout << lowval << " " << highval << endl;

		//saturate below the low percentile and above the high percentile
		tmpsplit[i].setTo(lowval, tmpsplit[i] < lowval);
		tmpsplit[i].setTo(highval, tmpsplit[i] > highval);

		//scale the channel
		cv::normalize(tmpsplit[i], tmpsplit[i], 0, 255, cv::NORM_MINMAX);
	}
	merge(tmpsplit, out);
}