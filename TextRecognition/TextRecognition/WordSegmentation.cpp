/*
*  This file contains functions that used to isolate different words
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

#include "WordSegmentation.h"

#include <opencv2\imgproc.hpp>

using namespace cv;

std::vector<int> averageDistanceByRow(cv::Mat & binary)
{
	std::vector<int> freq;
	int cols = binary.cols;
	for (int i = 0; i < binary.rows; i++)
	{
		uchar* row = binary.ptr<uchar>(i);
		freq.push_back(_rowAverageDistance(row, cols));
	}
	return freq;
}

int _rowAverageDistance(uchar * row, size_t size)
{
	int accumulate = 0;
	int count = 0;
	bool captured = false;
	int start = 0;
	int end;
	
	for (size_t i = 1; i < size; i++)
	{
		if (!captured && row[i] != 0 && row[i - 1] == 0)
		{
			captured = true;
			start = i;
		}
		if (captured && row[i] == 0)
		{
			captured = false;
			end = i;
			accumulate += end - start;
			++count;
		}
	}
	int result;
	if (count == 0)
	{
		result = 0;
	}
	else
	{
		result = accumulate / count;
	}
	return result;
}
