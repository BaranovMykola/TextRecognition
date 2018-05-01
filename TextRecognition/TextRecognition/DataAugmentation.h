#pragma once
#include <opencv2/core/mat.hpp>

std::vector<cv::Mat> dataAugmentation(const cv::Mat& sample);

double fRand(double fMin, double fMax);

void generateAffineMatrix(cv::Mat& affine);

std::vector<cv::Mat> affineAugmentation(const cv::Mat& sample, int count, int additionalElementsCount, double maxAdditionalElement, int background);

std::vector<cv::Mat> noiseAugmentation(const cv::Mat& sample, int blackCount, int whiteCount);

void fillRandomPixel(cv::Mat& sample, bool nonZero, float replace);