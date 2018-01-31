#include "WordSegmentation.h"

#include <opencv2\imgproc.hpp>

using namespace cv;

std::map<int, Spaces> segmentWords(cv::Mat & binary)
{
	std::map<int, Spaces> spaces;

	auto filled = fillLetters(binary);


	auto freqDistance = averageDistanceByRow(filled);

	/**/
	int m;
	int a;
	auto h = calculateProjectionHist(binary, &m,&a);
	auto hh = calculateGraphicHist(h,a);
	/**/

	auto linesPosition = extractLinesPosition(calculateProjectionHist(binary));
	auto chars = sortCharacters(binary.clone());

	bool cap = false;
	int s;
	int e;
	for (auto i : linesPosition)
	{
		uchar* row = filled.ptr<uchar>(i);
		spaces[i] = std::vector<int>();

		for (int j = 1; j < filled.cols; j++)
		{
			if (!cap && row[j] != 0 && row[j - 1] == 0)
			{
				cap = true;
				s = j;
			}
			else if (cap && row[j] == 0)
			{
				cap = false;
				e = j;
				auto diff = e - s;
				if (diff > freqDistance[i])
				{
					spaces[i].push_back(e);
					line(binary, Point(s, i), Point(e, i), Scalar::all(127), 10);
				}
			}
		}
	}

	return spaces;
}

std::vector<int> averageDistanceByRow(cv::Mat & binary)
{
	std::vector<int> freq;
	int cols = binary.cols;
	for (int i = 0; i < binary.rows; i++)
	{
		uchar* row = binary.ptr<uchar>(i);
		freq.push_back(_rowAverageDistance(row, cols));
	}
	return freq;
}

int _rowAverageDistance(uchar * row, size_t size)
{
	int accumulate = 0;
	int count = 0;
	bool captured = false;
	int start;
	int end;
	
	for (size_t i = 1; i < size; i++)
	{
		if (!captured && row[i] != 0 && row[i - 1] == 0)
		{
			captured = true;
			start = i;
		}
		if (captured && row[i] == 0)
		{
			captured = false;
			end = i;
			accumulate += end - start;
			++count;
		}
	}
	int result;
	if (count == 0)
	{
		result = 0;
	}
	else
	{
		result = accumulate / count;
	}
	return result;
}
