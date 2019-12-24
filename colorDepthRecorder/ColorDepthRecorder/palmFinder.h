#pragma once

#ifndef PALM_FINDER
#define PALM_FINDER

#define THRESHOLD 1
#define DIFF_THRESHOLD 5
#include <opencv2/opencv.hpp>
#include <algorithm>

inline bool sortcol(const std::vector<int>& v1,
	const std::vector<int>& v2) {
	return v1[0] < v2[0];
}

inline bool PalmFinder(cv::Mat source, cv::Point2f * massCenterLeft, cv::Point2f* massCenterRight, bool isTwohands = true)
{
	bool isPointValid = false;
	cv::Mat canny_output;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	// Detect edges using canny
	cv::Canny(source, canny_output, THRESHOLD, THRESHOLD, 3);
	// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	int numOfContours = contours.size();
	if (numOfContours > 0)
	{
		//std::vector<cv::Point2f> mc(contours.size());
		std::vector<std::vector<int>> areaVec(contours.size());
		std::vector<cv::Rect> boundRect(contours.size());
		std::vector<std::vector<cv::Point> > contours_poly(contours.size());
		// Get the moments
		std::vector<cv::Moments> mu(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			//mu[i] = moments(contours[i], false);
			approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
		}

		// Get the mass centers
		// and sorting 		
		for (int i = 0; i < contours.size(); i++)
			areaVec[i] = { (int)boundRect[i].area(),i };

		std::sort(areaVec.begin(), areaVec.end(), sortcol);
		/*std::cout << "The Matrix after sorting 1st row is:\n";
		for (int i = 0; i<areaVec.size(); i++)
		{
			for (int j = 0; j<areaVec[0].size(); j++)
				std::cout << areaVec[i][j] << " ";
			std::cout << std::endl;
		}*/

		if (isTwohands)
		{
			//checking if more than 2 points found
			cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
			cv::Scalar color = cv::Scalar(0, 0, 255);
			for (int i = 0; i< contours.size(); i++)
			{
				drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point());
				//circle(drawing, mc[i], 4, color, -1, 8, 0);
				cv::rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			}

			//cv::imshow("Contours", drawing);	

			if (numOfContours == 1)
			{
				massCenterLeft->x = boundRect[0].x;
				massCenterLeft->y = boundRect[0].y;
				massCenterRight->x = boundRect[0].x;
				massCenterRight->y = boundRect[0].y;
				isPointValid = true;
			}
			else // (numOfContours >= 2)
			{
				int biggestAreaMCIndex = areaVec[numOfContours - 1][1];
				int secondBiggestAreaMCIndex = areaVec[numOfContours - 2][1];
				int tempIndex = numOfContours - 2;
				while (
					abs(boundRect[secondBiggestAreaMCIndex].x - boundRect[biggestAreaMCIndex].x) < DIFF_THRESHOLD &&
					abs(boundRect[secondBiggestAreaMCIndex].y - boundRect[biggestAreaMCIndex].y) < DIFF_THRESHOLD)
				{
					tempIndex -= 1;
					secondBiggestAreaMCIndex = areaVec[tempIndex][1];
					if (secondBiggestAreaMCIndex < 0)
					{
						secondBiggestAreaMCIndex = 0;
						break;
					}
				}

				int halfWidth, halfHeight;
				if (boundRect[biggestAreaMCIndex].x < boundRect[secondBiggestAreaMCIndex].x) //loc1 == left
				{
					halfWidth = (int)floor(boundRect[biggestAreaMCIndex].width / 2.f);
					halfHeight = (int)floor(boundRect[biggestAreaMCIndex].height / 2.f);
					massCenterLeft->x = boundRect[biggestAreaMCIndex].x + halfWidth;
					massCenterLeft->y = boundRect[biggestAreaMCIndex].y + halfHeight;

					halfWidth = (int)floor(boundRect[secondBiggestAreaMCIndex].width / 2.f);
					halfHeight = (int)floor(boundRect[secondBiggestAreaMCIndex].height / 2.f);
					massCenterRight->x = boundRect[secondBiggestAreaMCIndex].x + halfWidth;
					massCenterRight->y = boundRect[secondBiggestAreaMCIndex].y + halfHeight;
				}
				else //loc2 == left
				{
					halfWidth = (int)floor(boundRect[secondBiggestAreaMCIndex].width / 2.f);
					halfHeight = (int)floor(boundRect[secondBiggestAreaMCIndex].height / 2.f);
					massCenterLeft->x = boundRect[secondBiggestAreaMCIndex].x + halfWidth;
					massCenterLeft->y = boundRect[secondBiggestAreaMCIndex].y + halfHeight;

					halfWidth = (int)floor(boundRect[biggestAreaMCIndex].width / 2.f);
					halfHeight = (int)floor(boundRect[biggestAreaMCIndex].height / 2.f);
					massCenterRight->x = boundRect[biggestAreaMCIndex].x + halfWidth;
					massCenterRight->y = boundRect[biggestAreaMCIndex].y + halfHeight;
				}

				isPointValid = true;
			}

			cv::imshow("Contours", drawing);
		}
		else
		{
			//checking if more than 2 points found
			cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
			cv::Scalar color = cv::Scalar(0, 0, 255);
			for (int i = 0; i< contours.size(); i++)
			{
				drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point());
				//circle(drawing, mc[i], 4, color, -1, 8, 0);
				cv::rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			}

			if (numOfContours == 1)
			{
				massCenterLeft->x = boundRect[0].x;
				massCenterLeft->y = boundRect[0].y;
				massCenterRight->x = boundRect[0].x;
				massCenterRight->y = boundRect[0].y;
				isPointValid = true;
			}
			else // (numOfContours >= 2)
			{
				int biggestAreaMCIndex = areaVec[numOfContours - 1][1];
				
				int halfWidth, halfHeight;
				halfWidth = (int)floor(boundRect[biggestAreaMCIndex].width / 2.f);
				halfHeight = (int)floor(boundRect[biggestAreaMCIndex].height / 2.f);
				massCenterLeft->x = boundRect[biggestAreaMCIndex].x + halfWidth;
				massCenterLeft->y = boundRect[biggestAreaMCIndex].y + halfHeight;
				massCenterRight->x = boundRect[biggestAreaMCIndex].x + halfWidth;
				massCenterRight->y = boundRect[biggestAreaMCIndex].y + halfHeight;				

				isPointValid = true;
			}

			cv::imshow("Contours", drawing);
		}
	
	}

	return isPointValid;
}

#endif