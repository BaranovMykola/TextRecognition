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
#include <map>

/**
 * \brief Calculate vertical projection histogram - count of zero pixels in each row
 * \param binary Binary image
 * \param min Minimum histogram value (optional)
 * \param max Maximum histogram value (optional)
 * \return Returns vertical projection histogram
 */
std::vector<int> calculateProjectionHist(cv::Mat& binary, int* min = 0, int* max = 0);

/**
 * \brief Makes visual histogram
 * \param freq Vertical projection histogram. See calculateProjectionHist(cv::Mat&)
 * \param maxFreq Maximum histogram value
 * \param bins Width of histogram visualization
 * \return Returns visualized histogram
 */
cv::Mat calculateGraphicHist(std::vector<int> freq, int maxFreq, int bins = 300);

/**
 * \brief Rotate image with croping size
 * \param source Image to rotate
 * \param angle Angle to rotate at
 * \return Returns rotated image
 */
cv::Mat rotate(cv::Mat& source, int angle);

/**
 * \brief Calculate skew angle
 * \param binary Binary image
 * \return Returns skew angle in degrees
 */
int findSkew(cv::Mat binary);

/**
 * \brief Try newAngle of skew. Rewrite angle and maxDev variables if newAngle is better
 * \param angle Rough skew angle
 * \param newAngle Try skew angle
 * \param resizedImage Binary images. Low resolution desirable
 * \param maxDev Rough maximal deviation
 */
void tryAngle(int& angle, int newAngle, cv::Mat& resizedImage, long long& maxDev);

/**
 * \brief Detects lines positions
 * \param binary Binary image of text
 * \return Return vector of exact lines positions
 */
std::vector<int> detectLines(cv::Mat& binary);

/**
 * \brief Converts lines ranges to exat line position
 * \param threshFreq Thresholded horizontal projection histogram
 * \param max Value to find ranges
 * \return Return vector of average position of each group
 */
std::vector<int> convertFreqToLines(std::vector<int> threshFreq, int max);

/**
 * \brief Clears lines duplicates. Used average distance between lines.
 * If distance less than half of average distance, lines replaces by lines of their
 * average position
 * \param lines Vector of lines position
 * \param binary Binary image of text
 * \return Returns lines without duplicates
 */
std::vector<int> clearMultipleLines(std::vector<int> lines, cv::Mat& binary);

/**
 * \brief Finds all letters that intersects exact line
 * \param line Line vertical position (Y)
 * \param binary Binsry image of text
 * \return Returns vector of letters position
 */
std::vector<cv::Rect> segmentExactLine(int line, cv::Mat& binary);