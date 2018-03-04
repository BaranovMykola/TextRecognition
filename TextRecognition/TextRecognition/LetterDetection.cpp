/*
*  This file contains functions that used to isolate letters from background
*  Copyright (C) 2018 Mykola Baranov
*
*  This program is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "LetterDetection.h"
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <vector>
#include <numeric>
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

int averLetterHight(cv::Mat& bin)
{
	auto letterRegions = encloseLetters(bin);
	return std::accumulate(letterRegions.begin(), letterRegions.end(), 0.0, [](double acc, Rect r) { return r.height + acc; }) / letterRegions.size();
}

cv::Mat closeCharacters(cv::Mat& binary)
{
	Mat closed;
	auto cloneBinary = binary.clone();
	int closingSize = averLetterHight(cloneBinary);
	auto kern = getStructuringElement(MORPH_ELLIPSE, Size(1, closingSize / 3 + 1));
	morphologyEx(binary, closed, MORPH_OPEN, kern, Point(0, closingSize / 3));
	return closed;
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

