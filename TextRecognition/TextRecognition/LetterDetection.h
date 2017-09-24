#pragma once
#include <opencv2\core.hpp>
#include <vector>

cv::Mat letterHighligh(cv::Mat & img);

std::vector<cv::Rect> encloseLetters(cv::Mat & thresholded);

std::vector<cv::Rect> filterRectangles(std::vector<std::vector<cv::Point>> contours);

void extractLetters(std::vector<cv::Rect> rects, cv::Mat& source);
