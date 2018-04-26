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

#pragma once
#include <opencv2\core.hpp>
#include <vector>

/**
 * \brief Detects lines positions
 * \param binary Binary image of text
 * \return Return vector of exact lines positions
 */
std::vector<int> detectLines(cv::Mat& binary);
//
///**
// * \brief Converts lines ranges to exat line position
// * \param threshFreq Thresholded horizontal projection histogram
// * \param max Value to find ranges
// * \return Return vector of average position of each group
// */
//std::vector<int> convertFreqToLines(std::vector<int> threshFreq, int max);

/**
 * \brief Clears lines duplicates. Used average distance between lines.
 * If distance less than half of average distance, lines replaces by lines of their
 * average position
 * \param lines Vector of lines position
 * \param binary Binary image of text
 * \return Returns lines without duplicates
 */
std::vector<int> clearMultipleLines(std::vector<int> lines, cv::Mat& binary);

///**
// * \brief Finds all letters that intersects exact line
// * \param line Line vertical position (Y)
// * \param binary Binsry image of text
// * \return Returns vector of letters position
// */
//std::vector<cv::Rect> segmentExactLine(int line, cv::Mat& binary);

///**
// * \brief Extract letter that lies on exact line
// * \param line Line number
// * \param allLetters Boundg rectangles of all letters
// * \param shift Vertical letters shift
// * \return Return bounding rectangles of letter that lies on exact line
// */
//std::vector<cv::Rect> _segmentExactLine(int line, std::vector<cv::Rect> allLetters, int shift);

/**
 * \brief Sort letters by lines
 * \param lines Vector of lines Y position
 * \param rects Letters bounding boxes
 * \param shift Bounding boxes Y shfting (in compare to original image)
 * \return Returns bounding box sorted by lines (1st dimentional - lines)
 */
std::vector<std::vector<cv::Rect>> segmentAllLines(std::vector<int> lines, std::vector<cv::Rect> rects, int shift);

/**
 * \brief Sort letters by lines
 * \param binary Binary image
 * \param lines Vector of lines position
 * \return Returns bounding box sorted by lines (1st dimentional - lines)
 */
std::vector<std::vector<cv::Rect>> segmentAllLines(cv::Mat& binary, std::vector<int> lines);