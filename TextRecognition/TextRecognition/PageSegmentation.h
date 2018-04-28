#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "TrainANN.h"

class PageSegmentation
{
public:
	explicit PageSegmentation(cv::Mat& original);

	void detect(cv::Ptr<cv::ml::ANN_MLP> mlp);
protected:
	int predict(cv::Ptr<cv::ml::ANN_MLP> mlp, cv::Mat& sample) const;

	std::vector<int> lines;
	std::vector<std::vector<cv::Rect>> sortedLetters;
	std::vector<std::vector<int>> spaces;
	cv::Mat binary;
};

