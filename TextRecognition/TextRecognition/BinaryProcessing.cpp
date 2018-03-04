/*
*  This file contains frequency used function in text recognition
*  Copyright (C) 2018 Mykola Baranov
*
*  This program is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "BinaryProcessing.h"

#include <numeric>

#include "LineSegmentation.h"
#include "LetterDetection.h"

cv::Mat fillLetters(cv::Mat & binary)
{
	auto chars = encloseLetters(binary);
	auto copy = binary.clone();
	for (auto i : chars)
	{
		copy(i) = 0;
	}
	return copy;
}

std::vector<int> extractLinesPosition(std::vector<int> freq)
{
	std::vector<int> linesPoistion;

	freq = thresholdLines(freq);

	int max = *std::max_element(freq.begin(), freq.end());

	bool captured = false;
	int strart = 0;
	int end;
	for (unsigned int i = 1; i < freq.size(); i++)
	{
		if (!captured && freq[i] == max && freq[i - 1] != max)
		{
			captured = true;
			strart = i;
		}
		else if (captured && freq[i] != max)
		{
			captured = false;
			end = i;
			linesPoistion.push_back((strart + end) / 2);
		}
	}

	return linesPoistion;
}

std::vector<int> thresholdLines(std::vector<int> freq)
{
	double average = std::accumulate(freq.begin(), freq.end(), 0) / (double)freq.size();
	auto minmax = std::minmax_element(freq.begin(), freq.end());
	double thresholdLevel = (average + *minmax.first) / 2;
	threshold(freq, thresholdLevel, *minmax.second);

	return freq;
}

void threshold(std::vector<int>& freq, int t, int max)
{
	for (auto& i : freq)
	{
		i = i >= t ? max : i;
	}
}