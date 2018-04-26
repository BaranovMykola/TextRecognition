/*
*  This file contains frequency used function in text recognition
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

#include "BinaryProcessing.h"

#include <numeric>

#include "LineSegmentation.h"
#include "LetterDetection.h"
#include <opencv2/imgproc.hpp>
#include "Contants.h"

using namespace cv;
using namespace std;

cv::Mat mat::fillLetters(cv::Mat & binary)
{
	auto chars = encloseLetters(binary);
	auto copy = binary.clone();
	for (auto i : chars)
	{
		copy(i) = 0;
	}
	return copy;
}

std::vector<int> mat::extractLinesPosition(std::vector<int> freq)
{
	std::vector<int> linesPoistion;

	freq = thresholdLines(freq);

	int max = *std::max_element(freq.begin(), freq.end());

	bool captured = false;
	int strart = 0;
	int end;
	for (unsigned int i = 1; i < freq.size(); i++)
	{
		if (!captured && freq[i] == max && freq[i - 1] != max)
		{
			captured = true;
			strart = i;
		}
		else if (captured && freq[i] != max)
		{
			captured = false;
			end = i;
			linesPoistion.push_back((strart + end) / 2);
		}
	}

	return linesPoistion;
}

std::vector<int> mat::thresholdLines(std::vector<int> freq)
{
	double average = std::accumulate(freq.begin(), freq.end(), 0) / (double)freq.size();
	auto minmax = std::minmax_element(freq.begin(), freq.end());
	double thresholdLevel = (average + *minmax.first) / 2;
	threshold(freq, static_cast<int>(thresholdLevel), *minmax.second);

	return freq;
}

void mat::threshold(std::vector<int>& freq, int t, int max)
{
	for (auto& i : freq)
	{
		i = i >= t ? max : i;
	}
}

cv::Mat mat::lineMorphologyEx(cv::Mat& binary)
{
	cv::Mat processed;
	morphologyEx(binary, processed, cv::MORPH_ERODE, getStructuringElement(cv::MORPH_RECT, cv::Size(HORIZONTAL_LINE_MORPHOLOGY_KERNEL_SIZE, 1)));
	return processed;
}

std::vector<int> mat::calculateProjectionHist(cv::Mat& binary, int* min, int* max)
{
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
}

cv::Mat mat::rotate(cv::Mat& source, int angle)
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
