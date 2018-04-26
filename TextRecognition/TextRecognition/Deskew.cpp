#include "Deskew.h"
#include <opencv2/imgproc.hpp>
#include <numeric>
#include "BinaryProcessing.h"
#include "Contants.h"

using namespace cv;

int findSkew(cv::Mat binary)
{
	long long maxDev = 0;
	int angle = 0;
	float sizeModifier = std::min({ binary.cols / SkrewRestoringImageSize, binary.rows / SkrewRestoringImageSize });
	sizeModifier = sizeModifier >= 1 ? sizeModifier : 1;
	cv::Mat resizedImage;
	cv::resize(binary, resizedImage, Size(static_cast<int>(binary.cols / sizeModifier), static_cast<int>(binary.rows / sizeModifier)), 0, 0, INTER_NEAREST);

	for (int i = -90; i <= 90; i += 5)
	{
		tryAngle(angle, i, resizedImage, maxDev);
	}

	for (int i = angle - 4; i <= angle + 4 && i != 0; i++)
	{
		tryAngle(angle, i, resizedImage, maxDev);
	}

	binary = mat::rotate(binary, angle);
	return angle;
}

void tryAngle(int& angle, int newAngle, cv::Mat& resizedImage, long long& maxDev)
{
	int min;
	int max;
	cv::Mat thresh = mat::rotate(resizedImage, newAngle);
	auto freq = mat::calculateProjectionHist(thresh, &min, &max);

	int aver = std::accumulate(freq.begin(), freq.end(), 0);
	aver /= static_cast<int>(freq.size());

	long long dev = static_cast<long long>(std::accumulate(freq.begin(), freq.end(), 0.0, [&](double acc, int elem)
	{
		return acc + pow((aver - elem), 2);
	}));

	if (maxDev < dev)
	{
		maxDev = dev;
		angle = newAngle;
	}
}