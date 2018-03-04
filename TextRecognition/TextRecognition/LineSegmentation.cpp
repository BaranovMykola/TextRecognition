#include "LineSegmentation.h"
#include <algorithm>
#include <numeric>
#include <opencv2\imgproc.hpp>
#include <iostream>

#include "Contants.h"
#include "LetterDetection.h"
#include <list>

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
	long long maxDev = 0;
	int angle = 0;
	int min;
	int max;
	float sizeModifier = std::min({ binary.cols / SkrewRestoringImageSize, binary.rows / SkrewRestoringImageSize });
	sizeModifier = sizeModifier >= 1 ? sizeModifier : 1;
	Mat resizedImage;
	cv::resize(binary, resizedImage, Size(binary.cols/sizeModifier, binary.rows/sizeModifier),0,0,INTER_NEAREST);

	for (int i = -90; i <= 90; i+=5)
	{
		_tryAngle(angle, i, resizedImage, maxDev);
	}

	for (int i = angle-4; i <= angle+4 && i != 0; i++)
	{
		_tryAngle(angle, i, resizedImage, maxDev);
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

void _tryAngle(int& angle, int newAngle, cv::Mat& resizedImage, long long& maxDev)
{
	int min;
	int max;
	cv::Mat thresh = rotate(resizedImage, newAngle);
	auto freq = calculateProjectionHist(thresh, &min, &max);

	auto lines = segmentLines(freq, min, max);//

	int aver = std::accumulate(freq.begin(), freq.end(), 0);
	aver /= freq.size();

	double dev = accumulate(freq.begin(), freq.end(), 0.0, [&](double acc, int elem) { return acc + pow((aver - elem), 2); });

	if (maxDev < dev)
	{
		maxDev = dev;
		angle = newAngle;
	}
}

//void threshold(std::vector<int>& freq, int t, int max)
//{
//	for (auto& i : freq)
//	{
//		i = i >= t ? max : i;
//	}
//}

std::map<cv::Rect, int, RectComparator> sortCharacters(cv::Mat & binary)
{
	int min;
	int max;

	std::map<Rect, int, RectComparator> rows;

	auto copy = binary.clone();
	auto rectangles = encloseLetters(binary);

	for (auto i : rectangles)
	{
		copy(i) = Scalar::all(0);
	}

	std::vector<int> freq = calculateProjectionHist(copy, &min, &max);
	double average = std::accumulate(freq.begin(), freq.end(), 0) / (double)freq.size();
	double thresholdLevel = (average + min) / 2;
	threshold(freq, thresholdLevel, max);

	auto h = calculateGraphicHist(freq, max);

	int s;
	int e;
	bool cap = false;
	int line = 0;
	for (int i = 0; i < freq.size(); i++)
	{
		if (freq[i] == max && !cap)
		{
			s = i;
			cap = true;
		}
		if (freq[i] != max && cap)
		{
			cap = false;
			e = i;

			for (auto j : rectangles)
			{
				if (j.y <= s && j.y + j.height >= s
					||
					j.y <= e && j.y + j.height >= e
					||
					j.y >= s && j.y + j.height <= e
					||
					j.y < s && j.y + j.height > e)
				{
					binary(j) = 100;
					rows[j] = line;
				}
			}
			++line;

		}
	}

	return rows;
}

std::vector<int> detectLines(cv::Mat& binary)
{
	int min;
	int max;
	auto freq = calculateProjectionHist(binary, &min, &max);

	freq = thresholdLines(freq);

	auto h = calculateGraphicHist(freq, max);

	auto lines = convertFreqToLines(freq, max);

	for (auto line : lines)
	{
		cv::line(binary, Point(0, line), Point(binary.cols, line), Scalar::all(127), 3);
	}

	return lines;
}

std::vector<int> convertFreqToLines(std::vector<int> threshFreq, int max)
{
	auto lowerBound = threshFreq.begin();
	std::vector<int> lineList;
	while(lowerBound != threshFreq.end())
	{
		auto lBound = std::find(lowerBound, threshFreq.end(), max);
		auto uBound = std::find_if(lBound, threshFreq.end(), [&](auto i) {return i != max; });
		if(lBound == uBound)
		{
			break;
		}
		int startGroup = std::distance(threshFreq.begin(), lBound);
		int endGroup = startGroup + std::distance(lBound, uBound)-1;
		int line = (startGroup + endGroup) / 2;
		lineList.push_back(line);
		lowerBound = uBound;
	}
	return lineList;
}

std::vector<int> clearMultipleLines(std::vector<int> lines, cv::Mat& binary)
{
	std::vector<int> clearedLines;
	std::vector<std::vector < int >> merging(lines.size());

	int sum = 0;
	for (int i = 1; i < lines.size(); ++i)
	{
		sum += (lines[i] - lines[i - 1]);
	}
	sum /= lines.size();

	int l =0;
	for (int i = 0; i < lines.size()-1; ++i)
	{
		if(std::abs(lines[i+1] - lines[i]) < sum/2)
		{
			merging[l].push_back(lines[i]);
			merging[l].push_back(lines[i+1]);
		}
		else
		{
			merging[l].push_back(lines[i]);
			++l;
			merging[l].push_back(lines[i+1]);
		}
	}

	auto itRem = std::remove_if(merging.begin(), merging.end(), [](auto vec) {return vec.empty(); });
	merging.erase(itRem, merging.end());
	for (auto & lineSet : merging)
	{
		auto uniqIt = std::unique(lineSet.begin(), lineSet.end());
		lineSet.erase(uniqIt, lineSet.end());

		int aver = std::accumulate(lineSet.begin(), lineSet.end(), 0)/lineSet.size();
		clearedLines.push_back(aver);
	}

	return clearedLines;
}


