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
#include "BinaryProcessing.h"

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

	auto lines = convertFreqToLines(freq, max);

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


