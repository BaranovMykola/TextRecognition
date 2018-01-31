#include "BinaryProcessing.h"

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
