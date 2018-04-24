/*
*  This file contains functions that used to detect lines from image
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


#include "LineSegmentation.h"
#include <algorithm>
#include <numeric>
#include <opencv2\imgproc.hpp>

#include <list>

#include "Contants.h"
#include "Line.h"
#include "BinaryProcessing.h"
#include "LetterDetection.h"
#include "VectorProcessing.h"
#include <opencv2/highgui.hpp>

using namespace cv;

std::vector<int> calculateProjectionHist(cv::Mat & binary, int * min, int * max)
{
	std::vector<int> freq;
	int _min = binary.cols;
	int _max = 0;
	for (int i = 0; i < binary.rows; i++)
	{
		uchar* row = binary.ptr<uchar>(i);
		int nonZeroQuantity = static_cast<int>(std::count_if(row, row + binary.cols, [](uchar p)
		{
			return p == 0;
		}));
		freq.push_back(nonZeroQuantity);
		if (_min > nonZeroQuantity)
		{
			_min = nonZeroQuantity;
		}
		if (_max < nonZeroQuantity)
		{
			_max = nonZeroQuantity;
		}
	}

	if (min != nullptr)
	{
		*min = _min;
	}
	if (max != nullptr)
	{
		*max = _max;
	}

	return freq;
}

cv::Mat calculateGraphicHist(std::vector<int> freq, int maxFreq, int bins)
{
	Mat hist = Mat::zeros(Size(bins, static_cast<int>(freq.size())), CV_8UC1);
	for (unsigned int i = 0; i < freq.size(); i++)
	{
		line(hist, Point(0, i), Point(bins*freq[i] / maxFreq,i), Scalar::all(254));
	}
	return hist;
}

cv::Mat rotate(cv::Mat& source, int angle)
{
	auto center = Point2f(source.cols / float(2), source.rows / float(2));
	cv::Mat rotateMat = getRotationMatrix2D(center, angle, 1);
	auto rotRect = RotatedRect(center, source.size(), static_cast<float>(angle)).boundingRect();
	rotateMat.at<double>(0, 2) += rotRect.width / 2.0 - center.x;
	rotateMat.at<double>(1, 2) += rotRect.height / 2.0 - center.y;

	cv::Mat rotatedImg = Mat::zeros(source.size(), source.type());
	warpAffine(source, rotatedImg, rotateMat, rotRect.size() - Size(1, 1), cv::INTER_LINEAR, BORDER_CONSTANT, Scalar::all(255));
	return rotatedImg;
}

int findDev(std::vector<bool> lines)
{
	int max = 0;
	int min = static_cast<int>(lines.size());
	int dev = 0;
	for (unsigned int i = 0; i < lines.size()-1; i++)
	{
		if (lines[i] == lines[i + 1])
		{
			++dev;
		}
		else
		{
			if (dev > max)
			{
				max = dev;
			}
			if (dev < min)
			{
				min = dev;
			}
		}
	}
	return abs(min - max);
}

int findSkew(cv::Mat binary)
{
	long long maxDev = 0;
	int angle = 0;
	float sizeModifier = std::min({ binary.cols / SkrewRestoringImageSize, binary.rows / SkrewRestoringImageSize });
	sizeModifier = sizeModifier >= 1 ? sizeModifier : 1;
	Mat resizedImage;
	cv::resize(binary, resizedImage, Size(static_cast<int>(binary.cols / sizeModifier), static_cast<int>(binary.rows / sizeModifier)),0,0,INTER_NEAREST);

	for (int i = -90; i <= 90; i+=5)
	{
		tryAngle(angle, i, resizedImage, maxDev);
	}

	for (int i = angle-4; i <= angle+4 && i != 0; i++)
	{
		tryAngle(angle, i, resizedImage, maxDev);
	}

	binary = rotate(binary, angle);
	return angle;
}

void tryAngle(int& angle, int newAngle, cv::Mat& resizedImage, long long& maxDev)
{
	int min;
	int max;
	cv::Mat thresh = rotate(resizedImage, newAngle);
	auto freq = calculateProjectionHist(thresh, &min, &max);

	int aver = std::accumulate(freq.begin(), freq.end(), 0);
	aver /= static_cast<int>(freq.size());

	long long dev = static_cast<long long>(accumulate(freq.begin(), freq.end(), 0.0, [&](double acc, int elem)
	{
		return acc + pow((aver - elem), 2);
	}));

	if (maxDev < dev)
	{
		maxDev = dev;
		angle = newAngle;
	}
}

std::vector<int> detectLines(cv::Mat& binary)
{
	int min;
	int max;
	auto freq = calculateProjectionHist(binary, &min, &max);

	freq = thresholdLines(freq);

	auto h = calculateGraphicHist(freq, max);

#if _DEBUG
	Mat b;
	morphologyEx(binary, b, MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(40, 1)));

	freq = calculateProjectionHist(binary, &min, &max);
	auto hist = calculateGraphicHist(freq, max);
	namedWindow("Hist", CV_WINDOW_FREERATIO);
	imshow("Hist", hist);
	waitKey();

	int kernel = 20; // 2n+1
	for (int i = kernel; i < freq.size() - kernel; ++i)
	{
		freq[i] = std::accumulate(freq.begin() + i - kernel, freq.begin() + i + kernel, 0) / (kernel * 2 + 1);
	}

	max = *std::max_element(freq.begin(), freq.end());
	hist = calculateGraphicHist(freq, max);
	imshow("Hist", hist);
	waitKey();
#endif

	//auto lines = convertFreqToLines(freq, max);

	auto lines = vec::findLocalMaxima(freq);

#if _DEBUG
	Mat draw = binary.clone();
	for (auto line : lines)
	{
		cv::line(draw, Point(0, line), Point(binary.cols, line), Scalar(127), 5);
	}
#endif

	return lines;
}

std::vector<int> convertFreqToLines(std::vector<int> threshFreq, int max)
{
	auto lowerBound = threshFreq.begin();
	std::vector<int> lineList;
	while(lowerBound != threshFreq.end())
	{
		auto lBound = std::find(lowerBound, threshFreq.end(), max);
		auto uBound = std::find_if(lBound, threshFreq.end(), [&](auto i) {return i != max; });
		if(lBound == uBound)
		{
			break;
		}
		int startGroup = static_cast<int>(std::distance(threshFreq.begin(), lBound));
		int endGroup = startGroup + static_cast<int>(std::distance(lBound, uBound))-1;
		int line = (startGroup + endGroup) / 2;
		lineList.push_back(line);
		lowerBound = uBound;
	}
	return lineList;
}

std::vector<int> clearMultipleLines(std::vector<int> lines, cv::Mat& binary)
{
	std::vector<int> clearedLines;
	std::vector<std::vector < int >> merging(lines.size());

	int sum = 0;
	for (unsigned int i = 1; i < lines.size(); ++i)
	{
		sum += (lines[i] - lines[i - 1]);
	}
	sum /= static_cast<int>(lines.size());

	int l =0;
	for (unsigned int i = 0; i < lines.size()-1; ++i)
	{
		if(std::abs(lines[i+1] - lines[i]) < sum/2)
		{
			merging[l].push_back(lines[i]);
			merging[l].push_back(lines[i+1]);
		}
		else
		{
			merging[l].push_back(lines[i]);
			++l;
			merging[l].push_back(lines[i+1]);
		}
	}

	auto itRem = std::remove_if(merging.begin(), merging.end(), [](auto vec) {return vec.empty(); });
	merging.erase(itRem, merging.end());
	for (auto & lineSet : merging)
	{
		auto uniqIt = std::unique(lineSet.begin(), lineSet.end());
		lineSet.erase(uniqIt, lineSet.end());

		int aver = std::accumulate(lineSet.begin(), lineSet.end(), 0)/static_cast<int>(lineSet.size());
		clearedLines.push_back(aver);
	}

	return clearedLines;
}

std::vector<cv::Rect> segmentExactLine(int line, cv::Mat& binary)
{
	auto clone = binary.clone();
	int shift = averLetterHight(binary)/3-5+4;
	binary = closeCharacters(binary);
	auto allLetters = encloseLetters(binary);

	for (auto letter : allLetters)
	{
		binary(letter) = 0;
	}

	return _segmentExactLine(line, allLetters, shift);
}

std::vector<cv::Rect> _segmentExactLine(int line, std::vector<cv::Rect> allLetters, int shift)
{
	std::vector<cv::Rect> letters;

	for (auto ch : allLetters)
	{
		if (ch.y < line && ch.y + ch.height > line)
		{
			ch.y -= shift;
			letters.push_back(ch);
		}
	}

	std::sort(letters.begin(), letters.end(), [](auto l, auto r) {return l.x < r.x; });

	return letters;
}

int distance(cv::Rect rect, int line)
{
	auto rectCenter = rect.y + rect.height / 2;
	return std::abs(line - rectCenter);
}

std::vector<std::vector<cv::Rect>> _segmentAllLines(std::vector<int> lines, std::vector<cv::Rect> rects, int shift)
{
	std::vector<std::vector<cv::Rect>> sortedRects(lines.size());

	for (int i = 0; i < rects.size(); ++i)
	{
		int minDist = INT_MAX;
		int minLine = -1;
		rects[i].y -= shift;
		for (int j = 0; j < lines.size(); ++j)
		{
			auto dist = distance(rects[i], lines[j]);
			if(dist < minDist)
			{
				minDist = dist;
				minLine = j;
			}
		}

		sortedRects[minLine].push_back(rects[i]);
		minLine = -1;
	}

	return sortedRects;
}

std::vector<std::vector<cv::Rect>> segmentAllLines(cv::Mat& binary, std::vector<int> lines)
{
	auto clone = binary.clone();
	int shift = averLetterHight(binary) / 3 - 5 + 4;
	binary = closeCharacters(binary);
	auto allLetters = encloseLetters(binary);

	for (auto letter : allLetters)
	{
		binary(letter) = 0;
	}

	auto sorted = _segmentAllLines(lines, allLetters, shift);
//#if _DEBUG
//	demo::drawLines(lines, clone);
//	namedWindow("Img", CV_WINDOW_FREERATIO);
//	for (auto element : sorted)
//	{
//		for (auto rect : element)
//		{
//			auto d1 = distance(rect, lines[1]);
//			auto d2 = distance(rect, lines[2]);
//			clone(rect) = 0;
//		}
//		imshow("Img", clone);
//		waitKey();
//	}
//#endif
	return sorted;
}
