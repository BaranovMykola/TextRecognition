#include "VectorProcessing.h"

#include  <numeric>
#include "Contants.h"
#include <opencv2/imgproc.hpp>

std::vector<int> vec::findLocalMaxima(std::vector<int> freq)
{
	std::vector<int> localMax;
	std::vector<int> localMaxQueue;

	for (int i = 0; i < freq.size(); ++i)
	{
		if(checkMax(freq,i))
		{
			localMaxQueue.push_back(i);
		}
		else if(!localMaxQueue.empty())
		{
			int pos = std::round(std::accumulate(localMaxQueue.begin(), localMaxQueue.end(), 0) / (double)localMaxQueue.size());
			localMax.push_back(pos);
			localMaxQueue.clear();
		}
	}

	return localMax;
}

bool vec::checkMax(std::vector<int> freq, int elemIndex)
{
	bool rightNoGrater = true;
	bool leftNoGrater = true;
	int kernel = freq[elemIndex];
	
	for (int i = elemIndex; i < freq.size(); ++i)
	{
		if(kernel < freq[i])
		{
			rightNoGrater = false;
			break;
		}
		if (kernel > freq[i])
		{
			break;
		}
	}

	for (int i = elemIndex; i > 0; --i)
	{
		if (kernel < freq[i])
		{
			leftNoGrater = false;
			break;
		}
		if(kernel > freq[i])
		{
			break;
		}
	}

	return rightNoGrater & leftNoGrater;
}

std::vector<int> vec::blutHistogram(const std::vector<int>& freq)
{
	std::vector<int> bluredHistogram(freq.size());
	for (int i = HISTOGRAM_BLUR_KERNEL_SIZE; i < freq.size() - HISTOGRAM_BLUR_KERNEL_SIZE; ++i)
	{
		bluredHistogram[i] = std::accumulate(freq.begin() + i - HISTOGRAM_BLUR_KERNEL_SIZE, freq.begin() + i + HISTOGRAM_BLUR_KERNEL_SIZE, 0) / (HISTOGRAM_BLUR_KERNEL_SIZE * 2 + 1);
	}

	return bluredHistogram;
}

cv::Mat vec::calculateGraphicHist(std::vector<int> freq, int maxFreq, int bins)
{
	cv::Mat hist = cv::Mat::zeros(cv::Size(bins, static_cast<int>(freq.size())), CV_8UC1);
	for (unsigned int i = 0; i < freq.size(); i++)
	{
		line(hist, cv::Point(0, i), cv::Point(bins*freq[i] / maxFreq, i), cv::Scalar::all(254));
	}
	return hist;
}

int vec::distance(cv::Rect rect, int line)
{
	auto rectCenter = rect.y + rect.height / 2;
	return std::abs(line - rectCenter);
}
