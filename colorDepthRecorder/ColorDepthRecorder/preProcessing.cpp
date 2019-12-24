#include "preProcessing.h"
#include "palmFinder.h"

Preprocessing::Preprocessing(cv::Size resultSize, cv::Size blurSize, int dilationRadius, int erosionRadius, float mahalDistance, int depthMax)
{
	resultSize_ = resultSize;
	blurSize_ = blurSize;
	dilationRadius_ = dilationRadius;
	erosionRadius_ = erosionRadius;
	depthMax_ = depthMax;

	skinSegmentation_ = new MahalSkinSegmentation(mahalDistance);

	croppedColorLeft_ = cv::Mat::zeros(resultSize_, CV_8UC3);
	croppedColorRight_ = cv::Mat::zeros(resultSize_, CV_8UC3);

	croppedDepthLeft_ = cv::Mat::zeros(resultSize_, CV_8UC1);
	croppedDepthRight_ = cv::Mat::zeros(resultSize_, CV_8UC1);

	croppedDepthColorizedLeft_ = cv::Mat::zeros(resultSize_, CV_8UC1);
	croppedDepthColorizedRight_ = cv::Mat::zeros(resultSize_, CV_8UC1);

	halfImageWidth_ = (int)floor(resultSize.width / 2.f);
	halfImageHeight_ = (int)floor(resultSize.height / 2.f);

	alpha_ = 1 / 4.f;

	pastLeftX_ = 0;
	pastLeftY_ = 0;
	pastRightX_ = 0;
	pastRightY_ = 0;
}

Preprocessing::~Preprocessing() 
{
	if (skinSegmentation_)
		delete skinSegmentation_;
}

bool Preprocessing::preProcess(cv::Mat * colormat, cv::Mat* depthmat)
{
	cv::Size inputSize = colormat->size();

	cv::Mat colorCopy = colormat->clone();
	cv::Mat depthCopy = depthmat->clone();

	cv::Mat maskColor;
	cv::Mat maskDepthF;
	cv::Mat maskDepth;
	cv::Mat resultMask;

	cv::Point2f palmCenterLeft;
	cv::Point2f palmCenterRight;
	//get mask from color
	cv::GaussianBlur(colorCopy, colorCopy, blurSize_, 0);
	maskColor = skinSegmentation_->getMask(colorCopy);
	maskColor.convertTo(maskColor, CV_8U);

	//get mask from depth
	cv::GaussianBlur(depthCopy, depthCopy, blurSize_, 0);
	cv::threshold(depthCopy, maskDepth, depthMax_, 255, CV_THRESH_BINARY_INV);

	//combining both mask
	bitwise_and(maskDepth, maskColor, resultMask);
	dilation(resultMask, resultMask, dilationRadius_);
	erosion(resultMask, resultMask, erosionRadius_);

	//finding central moment
	bool isPointsValid = PalmFinder(resultMask, &palmCenterLeft, &palmCenterRight,false);

	pastLeftX_ = alpha_*(palmCenterLeft.x - pastLeftX_) + pastLeftX_;
	pastLeftY_ = alpha_*(palmCenterLeft.y - pastLeftY_) + pastLeftY_;

	pastRightX_ = alpha_*(palmCenterRight.x - pastRightX_) + pastRightX_;
	pastRightY_ = alpha_*(palmCenterRight.y - pastRightY_) + pastRightY_;

	//croping 
	if (isPointsValid)
	{
		cv::Mat copycolor = colormat->clone();
		cv::Mat copydepth = depthmat->clone();
		//left 
		{
			cv::Point left;

			left.x = (int)floor(pastLeftX_ - halfImageWidth_);
			left.y = (int)floor(pastLeftY_ - halfImageHeight_);

			if (left.x < 0) left.x = 0;
			if ((left.x + resultSize_.width) > inputSize.width) left.x = inputSize.width - resultSize_.width;

			if (left.y < 0)left.y = 0;
			if ((left.y + resultSize_.height) > inputSize.height) left.y = inputSize.height - resultSize_.height;

			cv::Rect handArealeft = cv::Rect(left, resultSize_);

			croppedColorLeft_ = copycolor(handArealeft);
			
			croppedDepthLeft_ = copydepth(handArealeft);
		}
		//right
		{
			cv::Point right;

			right.x = (int)floor(pastRightX_ - halfImageWidth_);
			right.y = (int)floor(pastRightY_ - halfImageHeight_);

			if (right.x < 0) right.x = 0;
			if ((right.x + resultSize_.width) > inputSize.width) right.x = inputSize.width - resultSize_.width;

			if (right.y < 0) right.y = 0;
			if ((right.y + resultSize_.height) > inputSize.height) right.y = inputSize.height - resultSize_.height;

			cv::Rect handAreaRight = cv::Rect(right, resultSize_);

			croppedColorRight_ = copycolor(handAreaRight);

			croppedDepthRight_ = copydepth(handAreaRight);
		}	

	}
	return isPointsValid;
}


bool Preprocessing::preProcessEqed(cv::Mat *colormat, cv::Mat *depthmat, cv::Mat *eqedDepthMat)
{
	cv::Size inputSize = colormat->size();

	cv::Mat colorCopy = colormat->clone();
	cv::Mat depthCopy = depthmat->clone();

	cv::Mat maskColor;
	cv::Mat maskDepthF;
	cv::Mat maskDepth;
	cv::Mat resultMask;

	cv::Point2f palmCenterLeft;
	cv::Point2f palmCenterRight;
	//get mask from color
	cv::GaussianBlur(colorCopy, colorCopy, blurSize_, 0);
	maskColor = skinSegmentation_->getMask(colorCopy);
	maskColor.convertTo(maskColor, CV_8U);

	//get mask from depth
	cv::GaussianBlur(depthCopy, depthCopy, blurSize_, 0);
	cv::threshold(depthCopy, maskDepth, depthMax_, 255, CV_THRESH_BINARY_INV);

	//combining both mask
	bitwise_and(maskDepth, maskColor, resultMask);
	dilation(resultMask, resultMask, dilationRadius_);
	erosion(resultMask, resultMask, erosionRadius_);

	//finding central moment
	bool isPointsValid = PalmFinder(resultMask, &palmCenterLeft, &palmCenterRight, false);

	pastLeftX_ = alpha_*(palmCenterLeft.x - pastLeftX_) + pastLeftX_;
	pastLeftY_ = alpha_*(palmCenterLeft.y - pastLeftY_) + pastLeftY_;

	pastRightX_ = alpha_*(palmCenterRight.x - pastRightX_) + pastRightX_;
	pastRightY_ = alpha_*(palmCenterRight.y - pastRightY_) + pastRightY_;

	//croping 
	if (isPointsValid)
	{
		cv::Mat copycolor = colormat->clone();
		cv::Mat copydepth = eqedDepthMat->clone();

		//left 
		{
			cv::Point left;

			left.x = (int)floor(pastLeftX_ - halfImageWidth_);
			left.y = (int)floor(pastLeftY_ - halfImageHeight_);

			if (left.x < 0) left.x = 0;
			if ((left.x + resultSize_.width) > inputSize.width) left.x = inputSize.width - resultSize_.width;

			if (left.y < 0)left.y = 0;
			if ((left.y + resultSize_.height) > inputSize.height) left.y = inputSize.height - resultSize_.height;

			cv::Rect handArealeft = cv::Rect(left, resultSize_);

			croppedColorLeft_ = copycolor(handArealeft);
			croppedDepthLeft_ = copydepth(handArealeft);

		}
		//right
		{
			cv::Point right;

			right.x = (int)floor(pastRightX_ - halfImageWidth_);
			right.y = (int)floor(pastRightY_ - halfImageHeight_);

			if (right.x < 0) right.x = 0;
			if ((right.x + resultSize_.width) > inputSize.width) right.x = inputSize.width - resultSize_.width;

			if (right.y < 0) right.y = 0;
			if ((right.y + resultSize_.height) > inputSize.height) right.y = inputSize.height - resultSize_.height;

			cv::Rect handAreaRight = cv::Rect(right, resultSize_);

			croppedColorRight_ = copycolor(handAreaRight);
			croppedDepthRight_ = copydepth(handAreaRight);
		}

	}
	return isPointsValid;
}