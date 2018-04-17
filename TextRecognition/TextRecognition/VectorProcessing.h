#pragma once
#include  <vector>

namespace vec
{
	std::vector<int> findLocalMaxima(std::vector<int> freq);

	bool checkMax(std::vector<int> freq, int elemIndex);
}