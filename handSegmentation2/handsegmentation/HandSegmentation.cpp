#include <opencv/cv.hpp>
#include <iostream>
//#include "SkinColorBasedSegmentation.h"
//#include "KMeansSegmentation.h"
//#include "MeanShiftSegmentation.h"
#include "MahalSkinSegmentation.h"
using namespace cv;
using namespace std;	

int main(int argc, char** argv)
{
	//Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat image = imread(".//..//Resources//humanhand.jpg");	

	if (image.empty())
	{
		printf("No image");
		return 0;
	}

	imshow("Image Source", image);
	
	MahalSkinSegmentation mahalSkin_;
	if (!mahalSkin_.fetchData(".//..//Resources//Skin_NonSkin.txt",true)) return 1;
	if (!mahalSkin_.calcVarianceAndMean()) return 1;

	//Mat maskImage = getSkinMaskHSV(&image);
	//Mat maskImage = getKmeansSegmentation(&image,4,10,0.01f,5);
	//Mat segmentedImage = getMeanShiftedSegmentation(&image, 10, 1);

	Mat maskImage = mahalSkin_.getMahalanobisDistImg(image);
	imshow("maskImage", maskImage);

	Mat sepImg[3],segmentedImg[3];
	split(image, sepImg);

	bitwise_and(sepImg[0], maskImage, segmentedImg[0]);
	bitwise_and(sepImg[1], maskImage, segmentedImg[1]);
	bitwise_and(sepImg[2], maskImage, segmentedImg[2]);

	vector<Mat> arrayToMerge;
	arrayToMerge.push_back(segmentedImg[0]);
	arrayToMerge.push_back(segmentedImg[1]);
	arrayToMerge.push_back(segmentedImg[2]);

	Mat resImage;
	merge(arrayToMerge, resImage);
	imshow("Result", resImage);

	waitKey();
	
	return 0;
}