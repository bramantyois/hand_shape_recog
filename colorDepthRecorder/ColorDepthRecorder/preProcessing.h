#pragma once

#ifndef PRE_PROCESSING
#define PRE_PROCESSING

#include "MahalSkinSegmentation.h"
#include "MorphologicalOperation.h"

//#define DEPTH_MASK_MAX 4
//
//#define RESULT_IMAGE_WIDTH 320
//#define RESULT_IMAGE_HEIGHT 240
//
//#define RESULT_IMAGE_HALF_WIDTH 160
//#define RESULT_IMAGE_HALF_HEIGHT 120
//
//#define DILATION_RADIUS 7
//#define EROSION_RADIUS 7
//
//#define GAUSS_BLUR_SIZE cv::Size(21,21)

class Preprocessing
{
public:
	Preprocessing(cv::Size resultSize, cv::Size blurSize, int dilationRadius, int erosionRadius, float mahalDistance, int depthMax);
	~Preprocessing();

	bool preProcess(cv::Mat * colormat, cv::Mat * depthmat);
	bool preProcessEqed(cv::Mat * colormat, cv::Mat * depthmat, cv::Mat *eqedDepthMat);

	cv::Mat getCroppedColorLeft() { return croppedColorLeft_; };
	cv::Mat getCroppedColorRight() { return croppedColorRight_; };

	cv::Mat getCroppedDepthLeft() { return croppedDepthLeft_; };
	cv::Mat getCroppedDepthRight() { return croppedDepthRight_; };
	
	cv::Mat getColorizedDepthLeft() { return croppedDepthColorizedLeft_; };
	cv::Mat getColorizedDepthRight() { return croppedDepthColorizedRight_; };

private:
	MahalSkinSegmentation * skinSegmentation_;

	cv::Mat croppedColorLeft_;
	cv::Mat croppedColorRight_;

	cv::Mat croppedDepthLeft_;
	cv::Mat croppedDepthRight_;

	cv::Mat croppedDepthColorizedLeft_;
	cv::Mat croppedDepthColorizedRight_;
	
	cv::Size resultSize_;
	cv::Size blurSize_;
	
	int dilationRadius_;
	int erosionRadius_;

	int depthMax_;
	
	float pastLeftX_;
	float pastLeftY_;

	float pastRightX_;
	float pastRightY_;

	int halfImageWidth_;
	int halfImageHeight_;

	float alpha_;
};

#endif