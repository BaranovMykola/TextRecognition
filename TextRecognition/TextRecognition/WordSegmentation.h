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

#pragma once
#include <opencv2\core.hpp>

#include <vector>
#include <numeric>

#include "LineSegmentation.h"
#include "BinaryProcessing.h"

typedef std::vector<int> Spaces;

/* @brief

Returns vector of average distance between black pixels of each lines

See below various examples of the averageDistanceByRow function call:
@code
	averageDistanceByRow(binImg); // calculate average distance between black points for each line(row)
	averageDistanceByRow(fillLetters(binImg)); // calculate average distance between letters for each line (row)
@endcode

@param binary Binary image
*/
std::vector<int> averageDistanceByRow(cv::Mat& binary);

/* @brief

Calculate average distance between black pixels for certain row

@param row Pointer to image row
@param size Length of row
*/
int _rowAverageDistance(uchar* row, size_t size);

std::vector<int> checkSpaces(std::vector<cv::Rect> lettersInRow);