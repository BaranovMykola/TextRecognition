#include "DataAugmentation.h"
#include <cstdlib>
#include <vector>
#include <opencv2/core/mat.hpp>
#include "Contants.h"
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


std::vector<cv::Mat> dataAugmentation(const cv::Mat& sample)
{
	std::vector<cv::Mat> samples{ sample };
	auto affine = affineAugmentation(sample, AFFINE_SAMPLES, AFFINE_ADDITIONAL_ELEMENTS, AFFINE_MAX_ELEMENT, 1);
	auto noice = noiseAugmentation(sample, NOICE_BLACK_COUNT, NOICE_WHITHE_COUNT);

	samples.insert(samples.end(), affine.begin(), affine.end());

	return samples;
}

double fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

void generateAffineMatrix(cv::Mat& affine)
{
	const int rows = 2;
	const int cols = 3;
	const int shiftColumnIndex = 2;

	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			if(c == r)
			{
				continue;;
			}

			double newElement = fRand(0, AFFINE_MAX_ELEMENT);
			if(c == shiftColumnIndex)
			{
				newElement = fRand(1, AFFINE_MAX_SHIFT*2) - AFFINE_MAX_SHIFT;
			}

			affine.at<float>(r, c) = newElement;
		}
	}
}

std::vector<cv::Mat> affineAugmentation(const cv::Mat& sample, int count, int additionalElementsCount, double maxAdditionalElement, int background)
{
	vector<Mat> newSamples;

	for (int i = 0; i < count; ++i)
	{
		cv::Mat affine(cv::Size(3, 2), CV_32F);
		affine = 0;
		affine.diag() = 1;
		generateAffineMatrix(affine);
		
		Mat newSample(sample.size(), sample.type());
		newSample = background;
		warpAffine(sample, newSample, affine, sample.size(), INTER_NEAREST, BORDER_CONSTANT, Scalar::all(1));
		Mat diff;
		absdiff(sample, newSample, diff);
		if(cv::countNonZero(diff) == 0)
		{
			--i;
			continue;;
		}

		newSample.push_back(newSample);
	}

	return newSamples;
}

std::vector<cv::Mat> noiseAugmentation(const cv::Mat& sample, int blackCount, int whiteCount)
{
	std::vector<cv::Mat> samples{ sample };

	int cotextBlackCount = NOICE_RANDOM_COUNT ? fRand(1, blackCount) : blackCount;
	int cotextWhiteCount = NOICE_RANDOM_COUNT ? fRand(1, whiteCount) : whiteCount;

	for (int i = 0; i < cotextWhiteCount; ++i)
	{
		auto newSample = sample.clone();
		fillRandomPixel(newSample, true, 0);
	}

	return samples;
}

void fillRandomPixel(cv::Mat& sample, bool nonZero, float replace)
{
	std::vector<cv::Point2i> locs;
	Mat conv(sample.size(), CV_8UC1);
	sample.convertTo(conv, CV_8UC1);
	//sample = conv;
	if (nonZero)
	{
		cv::findNonZero(conv, locs);
	}
	else
	{
		auto invert = conv.clone();
		bitwise_not(invert, invert, invert);
		cv::findNonZero(invert, locs);
	}

	auto replacePixel = locs[rand() % locs.size()];

	sample.at<float>(replacePixel) = replace;
}
