
#ifndef MORPHOLOGICAL_OPERATION
#define MORPHOLOGICAL_OPERATION

#include <opencv2/opencv.hpp>	

#define DILATION_TYPE cv::MORPH_ELLIPSE
#define EROSION_TYPE cv::MORPH_RECT

inline void dilation(cv::Mat inputImage,cv::Mat resultImage, int dilationRadius)
{
	cv::Size dilationSize(2 * dilationRadius + 1, 2 * dilationRadius + 1);
	cv::Point dilationCenterPoint(dilationRadius, dilationRadius);

	cv::Mat element = cv::getStructuringElement(DILATION_TYPE, dilationSize, dilationCenterPoint);

	cv::dilate(inputImage, resultImage, element);
}

inline void erosion(cv::Mat inputImage, cv::Mat resultImage, int erosionRadius)
{
	cv::Size erosionSize(2 * erosionRadius + 1, 2 * erosionRadius + 1);
	cv::Point erosionCenterPoint(erosionRadius, erosionRadius);

	cv::Mat element = cv::getStructuringElement(EROSION_TYPE, erosionSize, erosionCenterPoint);

	cv::erode(inputImage, resultImage, element);
}

#endif
