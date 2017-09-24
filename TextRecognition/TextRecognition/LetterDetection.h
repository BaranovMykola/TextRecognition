#pragma once
#include <opencv2\core.hpp>
#include <vector>

cv::Mat letterHighligh(cv::Mat & img);

void encloseLetters(cv::Mat & img, cv::Mat& source);

std::vector<cv::Rect> filterRectangles(std::vector<std::vector<cv::Point>> contours);
