#include "LetterDetection.h"
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <vector>
#include "Contants.h"

using namespace cv;

cv::Mat letterHighligh(cv::Mat& img) 
{
	Mat dst; 
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	int blurSize = 3;
	int sigmaX = 0;
	int closingSize = 0;
		Mat closed;
		Mat blured;
		GaussianBlur(gray, blured, Size(blurSize*2+1, blurSize*2+1), sigmaX);
		threshold(blured, dst, closingSize, 255, THRESH_BINARY| THRESH_OTSU);
		morphologyEx(dst, closed, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(closingSize+1, closingSize+1)));
	return closed;
}

std::vector<cv::Rect> encloseLetters(cv::Mat& thresholded)
{
	std::vector<std::vector<cv::Point>> contours;
	Mat edges;
	Canny(thresholded, edges, 1, 200, 3);
	Mat closed = edges;
	findContours(closed, contours, RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
	Mat draw = Mat::zeros(thresholded.size(), CV_8UC3);
	std::vector<Rect> rects;
	for (auto i : contours)
	{
		auto rect = boundingRect(i);
		if (rect.area() > 100)
		{
			rects.push_back(rect);
			rectangle(thresholded, rect, Scalar(0, 0, 255), 1);
		}
	}
	drawContours(thresholded, contours, -1, Scalar(0, 255, 0), 1);
	return rects;
}

void extractLetters(std::vector<cv::Rect> rects, cv::Mat& source)
{
	Mat letter;
	int c = 0;
	for (auto i : rects)
	{
		letter = source(i);
		cv::Mat res;
		resize(letter, res, Size(28, 28));
		Mat edg;
		Canny(res, edg, 100, 200);
		if(cv::countNonZero(edg) > 0)
			imwrite(TextDatasetPathPrefix + std::to_string(c++) + ".jpg", res);
	}
}
