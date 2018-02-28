#pragma once
#include <opencv2\core.hpp>
#include <vector>
#include <map>

#include "RectComparator.h"
#include "BinaryProcessing.h"

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
 * \brief Detects line based on vertical projection histogram. See calculateProjectionHist(cv::Mat&)
 * \param freq Vertical projection histogram. See calculateProjectionHist(cv::Mat&)
 * \param min Minimum histogram value
 * \param max Maximum histogram value
 * \return Returns vector contains information about line for each row. True - line, False - no line
 */
std::vector<bool> segmentLines(std::vector<int> freq, int min, int max);

void visualizeLines(cv::Mat& img, std::vector<bool> lines, int width);

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

int countLines(std::vector<bool> lines);

/**
 * \brief Try newAngle of skew. Rewrite angle and maxDev variables if newAngle is better
 * \param angle Rough skew angle
 * \param newAngle Try skew angle
 * \param resizedImage Binary images. Low resolution desirable
 * \param maxDev Rough maximal deviation
 */
void _tryAngle(int& angle, int newAngle, cv::Mat& resizedImage, long long& maxDev);

/**
 * \brief Distributes letters between lines
 * \param binary Binary image
 * \return Return pairs letter-line
 */
std::map<cv::Rect, int, RectComparator> sortCharacters(cv::Mat& binary);