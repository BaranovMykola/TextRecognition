#pragma once
#include <opencv2\core.hpp>

#include <vector>
#include <numeric>

#include "LineSegmentation.h"
#include "BinaryProcessing.h"

typedef std::vector<int> Spaces;

std::vector<int, Spaces> segmentWords(cv::Mat& binary);

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