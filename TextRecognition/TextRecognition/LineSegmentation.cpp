#include "LineSegmentation.h"
#include <algorithm>
#include <opencv2\imgproc.hpp>

using namespace cv;

std::vector<int> calculateProjectionHist(cv::Mat & binary, int * min, int * max)
{
	std::vector<int> freq;
	int _min = binary.cols;
	int _max = 0;
	for (int i = 0; i < binary.rows; i++)
	{
		uchar* row = binary.ptr<uchar>(i);
		int nonZeroQuantity = std::count_if(row, row + binary.cols, [](uchar p)
		{
			return p == 0;
		});
		freq.push_back(nonZeroQuantity);
		if (_min > nonZeroQuantity)
		{
			_min = nonZeroQuantity;
		}
		if (_max < nonZeroQuantity)
		{
			_max = nonZeroQuantity;
		}
	}

	if (min != nullptr)
	{
		*min = _min;
	}
	if (max != nullptr)
	{
		*max = _max;
	}

	return freq;
}

cv::Mat calculateGraphicHist(std::vector<int> freq, int maxFreq, int bins)
{
	Mat hist = Mat::zeros(Size(bins, freq.size()), CV_8UC1);
	for (int i = 0; i < freq.size(); i++)
	{
		line(hist, Point(0, i), Point(bins*freq[i] / maxFreq,i), Scalar::all(255));
	}
	return hist;
}
