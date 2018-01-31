#pragma once
#include <opencv2\core.hpp>

#include <vector>
#include <numeric>

#include "LetterDetection.h"

cv::Mat fillLetters(cv::Mat& binary);

std::vector<int> extractLinesPosition(std::vector<int> freq);

std::vector<int> thresholdLines(std::vector<int> freq);

void threshold(std::vector<int>& freq, int t, int max);