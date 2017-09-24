#include "LetterDetection.h"
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <vector>

using namespace cv;

cv::Mat letterHighligh(cv::Mat& img) 
{
	Mat dst; 
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	int blurSize = 0;
	int sigmaX = 0;
	int closingSize = 0;
	namedWindow("Panel");
	createTrackbar("BlurSize", "Panel", &blurSize, 20);
	createTrackbar("SigmaX", "Panel", &sigmaX, 20);
	createTrackbar("closingSize", "Panel", &closingSize, 60);
		Mat closed;
	while (waitKey(30) != 27)
	{
		Mat blured;
		GaussianBlur(gray, blured, Size(blurSize*2+1, blurSize*2+1), sigmaX);
		//imshow("blured", blured);
		//adaptiveThreshold(blured, dst, 100, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, closingSize*2+3, 0);
		threshold(blured, dst, closingSize, 255, THRESH_BINARY| THRESH_OTSU);
		namedWindow("dst", CV_WINDOW_FREERATIO);
		morphologyEx(dst, closed, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(closingSize+1, closingSize+1)));
		imshow("dst", closed);
		
	}
	return closed;
}

void encloseLetters(cv::Mat& img, cv::Mat& source) 
{
	std::vector<std::vector<cv::Point>> contours;
	Mat edges;
	Canny(img, edges, 1, 200, 3);
	Mat closed;
	morphologyEx(edges, closed, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	findContours(closed, contours, RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
	Mat draw = Mat::zeros(img.size(), CV_8UC3);
	for (auto i : contours)
	{
		auto rect = boundingRect(i);
		rectangle(draw, rect, Scalar(0, 0, 255), 1);
		rectangle(source, rect, Scalar(0, 0, 255), 4);
	}


	drawContours(draw, contours, -1, Scalar(0, 255, 0), 1);
	namedWindow("encloseLetters", CV_WINDOW_FREERATIO);
	imshow("encloseLetters", img);
}

std::vector<Rect> filterRectangles(std::vector<std::vector<cv::Point>> contours)
{
	std::vector<Rect> rects;
	for (auto i = contours.begin();i < contours.end();++i)
	{
		for (auto j = std::next(i);j < contours.end();++j)
		{
			auto rect1 = boundingRect(*i);
			auto rect2 = boundingRect(*j);
			if ((rect1 & rect2).area() > 0)
			{
				rects.push_back(rect1 | rect2);
			}
			else
			{
				rects.push_back(rect1);
				rects.push_back(rect2);
			}
		}
	}
	return rects;
}

