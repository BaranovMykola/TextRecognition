#pragma once
#include <opencv2\core.hpp>
#include <opencv2\ml.hpp>

cv::Mat convertMatToVec(const cv::Mat & mat);

cv::Ptr<cv::ml::TrainData> loadTrainData();

void trainANN(std::string saveTo);

cv::Ptr<cv::ml::ANN_MLP> loadANN(std::string file);

void testANN(cv::Ptr<cv::ml::ANN_MLP> mlp);

void printLetters();