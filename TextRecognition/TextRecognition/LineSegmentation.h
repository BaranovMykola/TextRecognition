#pragma once
#include <opencv2\core.hpp>
#include <vector>

std::vector<int> calculateProjectionHist(cv::Mat& binary, int* min = 0, int* max = 0);

cv::Mat calculateGraphicHist(std::vector<int> freq, int maxFreq, int bins = 300);