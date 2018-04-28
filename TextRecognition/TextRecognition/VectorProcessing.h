#pragma once
#include  <vector>
#include <opencv2/core/mat.hpp>

namespace vec
{
	std::vector<int> findLocalMaxima(std::vector<int> freq);

	bool checkMax(std::vector<int> freq, int elemIndex);

	std::vector<int> blutHistogram(const std::vector<int>& freq);

	/**
	* \brief Makes visual histogram
	* \param freq Vertical projection histogram. See calculateProjectionHist(cv::Mat&)
	* \param maxFreq Maximum histogram value
	* \param bins Width of histogram visualization
	* \return Returns visualized histogram
	*/
	cv::Mat calculateGraphicHist(std::vector<int> freq, int maxFreq, int bins = 300);

	/**
	 * \brief Calculate distance between rectangle and line
	 * \param rect Bounding box rectangle
	 * \param line Line position
	 * \return Return distance between rectangle and line
	 */
	int distance(cv::Rect rect, int line);

	int distance(cv::Rect rect1, cv::Rect rect2);

	int averageXDistance(std::vector<cv::Rect> letters);
}
