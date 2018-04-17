#include "VectorProcessing.h"

#include  <numeric>

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
