#include "MahalSkinSegmentation.h"
#include "MorphologicalOperation.h"

MahalSkinSegmentation::MahalSkinSegmentation(float mahalDistance) :
	rgbData_(NULL),mahalDistance_(mahalDistance)
{
	dataSet_.data = NULL;
	isIncovarMeanCalculated_ = false;
	
	invCovarMat_ = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
	meanVec_.zeros();

	//isDataValid_ = fetchData(".\\Skin_NonSkin.txt", true);
	//if (isDataValid_) calcVarianceAndMean();	
	calcVarianceAndMean();
}

MahalSkinSegmentation::~MahalSkinSegmentation()
{
	if (rgbData_) delete[] rgbData_;
}

bool MahalSkinSegmentation::fetchData(std::string fileAddress, bool useOnlyRealDataSet)
{
	useOnlyRealDataSet_ = useOnlyRealDataSet;

	std::string line;
	std::ifstream file(fileAddress);

	dataSet_.numOfData = 0;
	dataSet_.numOfRealData = 0;
	if (file.is_open())
	{
		while (std::getline(file, line)) dataSet_.numOfData++;

		if (dataSet_.numOfData > 0)
		{
			file.clear();
			file.seekg(0, std::ios::beg);

			rgbData_ = new UCIrvineRGB[dataSet_.numOfData];
			memset(rgbData_, 0, dataSet_.numOfData * sizeof(UCIrvineRGB));
			
			int dataIndex = 0;
			int realDataNum = 0;
			for (dataIndex; dataIndex < dataSet_.numOfData; dataIndex++)
			{
				std::getline(file, line);
				std::istringstream ss(line);
				int b, g, r, t;
				//ss >> rgbData_[dataIndex].colorData[0] >> rgbData_[dataIndex].colorData[1] >> rgbData_[dataIndex].colorData[2] >> rgbData_[dataIndex].type;
				ss >> b >> g >> r >> t;
				rgbData_[dataIndex].colorData[0] = (b);
				rgbData_[dataIndex].colorData[1] = (g);
				rgbData_[dataIndex].colorData[2] = (r);
				rgbData_[dataIndex].type = t;
				if (t == 1)	realDataNum++;
			}
			
			dataSet_.data = rgbData_;
			dataSet_.numOfRealData = realDataNum;

			isDataValid_ = true;
			return true;
		}
	}
	return false;
}

bool MahalSkinSegmentation::calcVarianceAndMean()
{
	/*if (isDataValid_)
	{
		cv::Mat data, meanVec, newCovarMat;
		if (useOnlyRealDataSet_)
		{
			data = cv::Mat::zeros(cv::Size(3, dataSet_.numOfRealData), CV_64F);

			int realDataIndex = 0;
			for (int i = 0; i < dataSet_.numOfData; i++)
			{				
				if (rgbData_[i].type == 1)
				{
					double b = (double)rgbData_[i].colorData[0];
					double g = (double)rgbData_[i].colorData[1];
					double r = (double)rgbData_[i].colorData[2];

					data.at<double>(realDataIndex, 0) = b;
					data.at<double>(realDataIndex, 1) = g;
					data.at<double>(realDataIndex, 2) = r;
					realDataIndex++;
				}
			}
			cv::calcCovarMatrix(data, newCovarMat, meanVec, CV_COVAR_NORMAL | CV_COVAR_ROWS);

			newCovarMat = newCovarMat / (dataSet_.numOfRealData - 1);
			invCovarMat_ = newCovarMat.clone().inv(cv::DECOMP_SVD);
		}
		else
		{
			cv::Mat data = cv::Mat::zeros(cv::Size(3, dataSet_.numOfData), CV_64F);
			
			for (int i = 0; i < dataSet_.numOfData; i++)
			{
				double b = (double)rgbData_[i].colorData[0];
				double g = (double)rgbData_[i].colorData[1];
				double r = (double)rgbData_[i].colorData[2];

				data.at<double>(i,0) = b;
				data.at<double>(i,1) = g;
				data.at<double>(i,2) = r;

				//meanVec[0] += b;
				//meanVec[1] += g;
				//meanVec[2] += r;
			}

			cv::calcCovarMatrix(data, newCovarMat, meanVec, CV_COVAR_NORMAL | CV_COVAR_ROWS);

			newCovarMat = newCovarMat / (dataSet_.numOfData - 1);
			invCovarMat_ = newCovarMat.clone().inv(cv::DECOMP_SVD);
		}	
	
		//meanVec = meanVec / (dataSet_.numOfData);
		std::printf("inv covar mat \n %.10f %.10f %.10f , %.10f %.10f %.10f , %.10f %.10f %.10f \r\n",
			invCovarMat_.at<double>(0, 0), invCovarMat_.at<double>(0, 1), invCovarMat_.at<double>(0, 2),
			invCovarMat_.at<double>(1, 0), invCovarMat_.at<double>(1, 1), invCovarMat_.at<double>(1, 2),
			invCovarMat_.at<double>(2, 0), invCovarMat_.at<double>(2, 1), invCovarMat_.at<double>(2, 2));
		
		meanVec_ = meanVec;
		float mean0 = meanVec_[0];
		float mean1 = meanVec_[1];
		float mean2 = meanVec_[2];
		std::printf("mean vec \n %.10f %.10f %.10f \r\n", mean0, mean1, mean2);
		isIncovarMeanCalculated_ = true;
		return true;
	}*/
	//harcoding 	
	invCovarMat_.at<double>(0, 0) = 0.0052049648 ;
	invCovarMat_.at<double>(0, 1) = -0.0093383744 ;
	invCovarMat_.at<double>(0, 2) =0.0037136186 ;

	invCovarMat_.at<double>(1, 0) = -0.0093383744 ;
	invCovarMat_.at<double>(1, 1) = 0.0265522980 ;
	invCovarMat_.at<double>(1, 2) = -0.0156005413 ;

	invCovarMat_.at<double>(2, 0) = 0.0037136186 ;
	invCovarMat_.at<double>(2, 1) =-0.0156005413 ;
	invCovarMat_.at<double>(2, 2) = 0.0115064251;

	meanVec_[0] = 113.8698730469;
	meanVec_[1] = 146.6011199951;
	meanVec_[2] = 203.9919433594;
	
	/*std::printf("inv covar mat \n %.10f %.10f %.10f , %.10f %.10f %.10f , %.10f %.10f %.10f \r\n",
			invCovarMat_.at<double>(0, 0), invCovarMat_.at<double>(0, 1), invCovarMat_.at<double>(0, 2),
			invCovarMat_.at<double>(1, 0), invCovarMat_.at<double>(1, 1), invCovarMat_.at<double>(1, 2),
			invCovarMat_.at<double>(2, 0), invCovarMat_.at<double>(2, 1), invCovarMat_.at<double>(2, 2));

	std::printf("mean vec \n %.10f %.10f %.10f \r\n", meanVec_[0], meanVec_[1], meanVec_[2]);*/
	isIncovarMeanCalculated_ = true;

	return false;
}

void MahalSkinSegmentation::setMahalDistance(float mahalDist)
{
	mahalDistance_ = mahalDist;
}

cv::Mat MahalSkinSegmentation::getMahalanobisDistImg(const cv::Mat sourceImg)
{
	int frameWidth = sourceImg.size().width;
	int frameHeight = sourceImg.size().height;
	cv::Mat resultMat = cv::Mat::zeros(cv::Size(frameWidth, frameHeight), CV_64F);
	if (isIncovarMeanCalculated_ && (sourceImg.channels() == 3))
	{
		for (int y = 0; y < frameWidth; y++)
		{
			for (int x = 0; x < frameHeight; x++)
			{
				cv::Vec3d pixel = sourceImg.at<cv::Vec3b>(x,y);

				double mahalDistance = (double)cv::Mahalanobis(pixel, meanVec_, invCovarMat_);
				resultMat.at<double>(x, y) = mahalDistance;
			}
		}
	}
	return resultMat;
}

cv::Mat MahalSkinSegmentation::getMask(cv::Mat sourceImg)
{
	cv::Mat mahalDistImg = getMahalanobisDistImg(sourceImg);
	cv::Mat mask;
	cv::threshold(mahalDistImg, mask, mahalDistance_, 255, CV_THRESH_BINARY_INV);
	return mask;
}