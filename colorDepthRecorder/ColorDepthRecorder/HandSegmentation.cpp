#include "handSegmentation.h"

HandSegmentation::HandSegmentation(cv::Size blurSize, int dilationRadius, int erosionRadius, float mahalDistance)
{
	blurSize_ = blurSize;
	dilationRadius_ = dilationRadius;
	erosionRadius_ = erosionRadius;

	skinSegmentation_ = new MahalSkinSegmentation(mahalDistance);

	croppedColor_= cv::Mat::zeros(resultSize_, CV_8UC3);
}

HandSegmentation::~HandSegmentation()
{
	if (skinSegmentation_)
		delete skinSegmentation_;
}

void HandSegmentation::process(cv::Mat * colormat, cv::Mat * depthmat)
{
	cv::Mat colorCopy = colormat->clone();
	cv::Mat depthCopy = depthmat->clone();
	cv::Mat mask;
	cv::Mat sepColorImg[3], segmentedColorImg[3];

	//get mask from color
	cv::GaussianBlur(colorCopy, colorCopy, blurSize_, 0);
	mask = skinSegmentation_->getMask(colorCopy);
	dilation(mask, mask, dilationRadius_);
	erosion(mask, mask, erosionRadius_);

	cv::split(*colormat, sepColorImg);
	bitwise_and(sepColorImg[0], mask, segmentedColorImg[0]);
	bitwise_and(sepColorImg[1], mask, segmentedColorImg[1]);
	bitwise_and(sepColorImg[2], mask, segmentedColorImg[2]);
	
	std::vector<cv::Mat> arrayToMerge;
	arrayToMerge.push_back(sepColorImg[0]);
	arrayToMerge.push_back(sepColorImg[1]);
	arrayToMerge.push_back(sepColorImg[2]);
	merge(arrayToMerge, croppedColor_);

}