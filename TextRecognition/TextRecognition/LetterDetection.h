#pragma once
#include <opencv2\core.hpp>
#include <vector>

/**
 * \brief Thresholded image in order to detect letters
 * \param img Source image
 * \return Returns binary image. White - background. Black - letter
 */
cv::Mat letterHighligh(cv::Mat & img);

/**
 * \brief Enclose all letters
 * \param thresholded Binary image
 * \return Returns vector of rectangles contains each letter 
 */
std::vector<cv::Rect> encloseLetters(cv::Mat & thresholded);

/**
 * \brief Calculate average hight of letters
 * \param bin Binary image
 * \return Return average hight of letters
 */
int averLetterHight(cv::Mat & bin);

/**
 * \brief Connect splitted letters using vertical morphology operation
 * \param binary Binary image
 * \return Return binary image with connected letters.
 */
cv::Mat closeCharacters(cv::Mat& binary);

/**
 * \brief Write all letters to separate files
 * \param rects Letters rectangles
 * \param source Source image
 */
void extractLetters(std::vector<cv::Rect> rects, cv::Mat& source);
