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

#pragma once
#include <opencv2\core.hpp>

#include <vector>


namespace mat
{
	/**
	 * \brief Fill all letters in black
	 * \param binary Binary image
	 * \return Returns images with filled letters in black color
	 */
	cv::Mat fillLetters(cv::Mat& binary);

	/**
	 * \brief Calculate exact lines position
	 * \param freq Vertical projection histogram. See calculateProjHist(cv::Mat&)
	 * \return Return vector of exact lines position
	 */
	std::vector<int> extractLinesPosition(std::vector<int> freq);

	/**
	 * \brief Detects lines range from vertical projection histogram
	 * \param freq Vertical projection histogram. See calculateProjHist(cv::Mat&)
	 * \return Returns thresholded histogram. Thresholded position means line
	 */
	std::vector<int> thresholdLines(std::vector<int> freq);

	/**
	 * \brief Thresholded histogram
	 * \param freq Vertical projection histogram. See calculateProjHist(cv::Mat&)
	 * \param t Threshold level
	 * \param max Values to replace values higher than threshold level
	 */
	void threshold(std::vector<int>& freq, int t, int max);

	/**
	 * \brief Apply morphology operation for improoving line detection accuracy
	 * \param binary Binary image with thresholded lines
	 * \return Return binary image with applied morphlogy operations
	 */
	cv::Mat lineMorphologyEx(cv::Mat& binary);

	/**
	* \brief Calculate vertical projection histogram - count of zero pixels in each row
	* \param binary Binary image
	* \param min Minimum histogram value (optional)
	* \param max Maximum histogram value (optional)
	* \return Returns vertical projection histogram
	*/
	std::vector<int> calculateProjectionHist(cv::Mat& binary, int* min = 0, int* max = 0);

	/**
	* \brief Rotate image with croping size
	* \param source Image to rotate
	* \param angle Angle to rotate at
	* \return Returns rotated image
	*/
	cv::Mat rotate(cv::Mat& source, int angle);
}