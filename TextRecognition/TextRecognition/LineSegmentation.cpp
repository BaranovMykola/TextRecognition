#include "LineSegmentation.h"
#include <algorithm>
#include <numeric>
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
		line(hist, Point(0, i), Point(bins*freq[i] / maxFreq,i), Scalar::all(254));
	}
	return hist;
}

std::vector<bool> segmentLines(std::vector<int> freq, int min, int max)
{
	std::vector<bool> lines;
	for (auto i : freq)
	{
		if (abs(i - max) > abs(i - min))
		{
			//line
			lines.push_back(true);
		}
		else
		{
			//space
			lines.push_back(false);
		}
	}
	return lines;
}

void visualizeLines(cv::Mat & img, std::vector<bool> lines, int width)
{
	for (int i = 0; i < img.rows && i < lines.size(); i++)
	{
		Scalar color;
		if (lines[i])
		{
			color = Scalar::all(0);
		}
		else
		{
			color = Scalar::all(255);
		}

		line(img, Point(0, i), Point(width, i), color);
	}
}

cv::Mat rotate(cv::Mat& source, int angle)
{
	auto center = Point2f(source.cols / 2, source.rows / 2);
	cv::Mat rotateMat = getRotationMatrix2D(center, angle, 1);
	auto rotRect = RotatedRect(center, source.size(), angle).boundingRect();
	rotateMat.at<double>(0, 2) += rotRect.width / 2.0 - center.x;
	rotateMat.at<double>(1, 2) += rotRect.height / 2.0 - center.y;

	cv::Mat rotatedImg = Mat::zeros(source.size(), source.type());
	warpAffine(source, rotatedImg, rotateMat, rotRect.size() - Size(1, 1), cv::INTER_LINEAR, BORDER_CONSTANT, Scalar::all(255));
	return rotatedImg;
}

int findDev(std::vector<bool> lines)
{
	int max = 0;
	int min = lines.size();
	int dev = 0;
	for (int i = 0; i < lines.size()-1; i++)
	{
		if (lines[i] == lines[i + 1])
		{
			++dev;
		}
		else
		{
			if (dev > max)
			{
				max = dev;
			}
			if (dev < min)
			{
				min = dev;
			}
		}
	}
	return abs(min - max);
}

int findSkew(cv::Mat binary)
{
	int maxDev = 0;
	int angle = 0;
	int min;
	int max;

	for (int i = 0; i < 360; i++)
	{
		cv::Mat thresh = rotate(binary, i);
		auto freq = calculateProjectionHist(thresh, &min, &max);
		//auto hist = calculateGraphicHist(freq, max);

		auto lines = segmentLines(freq, min, max);
		//visualizeLines(thresh, lines, 300);



		int aver = std::accumulate(freq.begin(), freq.end(), 0);
		aver /= freq.size();

		double dev = accumulate(freq.begin(), freq.end(), 0.0, [&](double acc, int elem) { return acc + pow((aver - elem), 2); });

		if (maxDev < dev)
		{
			maxDev = dev;
			angle = i;
		}

		//if (minLines > countLines(lines))
		//{
		//	minLines = countLines(lines); // ---------- opt
		//	angle = i;
		//}

		//std::vector<int> distr(11);
		////int mod = freq.size() / 10;
		//for (int j = 0; i < freq.size(); j++)
		//{

		//	distr[j / (freq.size() / 10)]+=freq[j];
		//}

	}
	binary = rotate(binary, angle);
	return angle;
}

int countLines(std::vector<bool> lines)
{
	int count = 0;
	for (int i = 0; i < lines.size() - 1; i++)
	{
		if (lines[i] != lines[i + 1])
		{
			++count;
		}
	}
	return count;
}
