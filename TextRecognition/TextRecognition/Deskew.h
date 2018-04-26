#pragma once

#include <opencv2/core.hpp>

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