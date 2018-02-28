#pragma once
#include <opencv2\core.hpp>

#include <vector>

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