#pragma once
#include <opencv2\core.hpp>
#include <vector>
#include <map>

#include "RectComparator.h"

std::vector<int> calculateProjectionHist(cv::Mat& binary, int* min = 0, int* max = 0);

cv::Mat calculateGraphicHist(std::vector<int> freq, int maxFreq, int bins = 300);

std::vector<bool> segmentLines(std::vector<int> freq, int min, int max);

void visualizeLines(cv::Mat& img, std::vector<bool> lines, int width);

cv::Mat rotate(cv::Mat& source, int angle);

int findSkew(cv::Mat binary);

int countLines(std::vector<bool> lines);

void _tryAngle(int& angle, int newAngle, cv::Mat& resizedImage, long long& maxDev);

std::map<cv::Rect, int, RectComparator> sortCharacters(cv::Mat& binary, std::vector<int> freq);