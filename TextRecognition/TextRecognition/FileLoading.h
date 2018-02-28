#pragma once

#include <opencv2\core.hpp>

/**
 * \brief Load images from file to the image
 * \param img Image to load in
 * \param fileName Name of file (without extension). Loaded 'TestSamplePathPrefix+fileName+".jpg"'
 */
void loadImg(cv::Mat & img, const std::string & fileName);
