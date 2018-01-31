#include "WordSegmentation.h"

std::vector<int, Spaces> segmentWords(cv::Mat & binary)
{
	std::vector<int, Spaces> spaces;

	auto filled = fillLetters(binary);


	auto freqDistance = averageDistanceByRow(filled);

	auto linesPosition = extractLinesPosition(calculateProjectionHist(binary));

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
