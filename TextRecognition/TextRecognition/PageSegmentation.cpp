#include "PageSegmentation.h"
#include "WordSegmentation.h"
#include "Deskew.h"
#include "LetterDetection.h"
#include <opencv2/imgproc.hpp>

using namespace cv;

PageSegmentation::PageSegmentation(cv::Mat& original)
{
	auto thresh = letterHighligh(original);
	auto skew = findSkew(thresh);
	binary = mat::rotate(thresh, skew);
	auto filled = mat::fillLetters(binary);
	
	lines = detectLines(filled);
	sortedLetters = segmentAllLines(binary, lines);
	
	for (auto& lineLetters : sortedLetters)
	{
		std::sort(lineLetters.begin(), lineLetters.end(), [](auto l, auto r) {return l.x < r.x; });
		spaces.push_back(checkSpaces(lineLetters));
	}
}

void PageSegmentation::detect(cv::Ptr<cv::ml::ANN_MLP> mlp)
{
	setlocale(LC_CTYPE, "ukr");
	std::string letters = "ÀàÁáÂâÃãÄäÅåªºÆæÇçÈè²³¯¿ÉéÊêËëÌìÍíÎîÏïĞğÑñÒòÓóÔôÕõÖö×÷ØøÙùÜüŞşßÿ";

	int j = 0;
	for (auto lineLetter : sortedLetters)
	{
		int s = 0;
		for (int i = 0; i < lineLetter.size(); ++i)
		{
			if (s < spaces[j].size() && spaces[j][s] == i)
			{
				std::cout << " ";
				if (s < spaces.size() - 1)
				{
					++s;
				}
			}
			Mat sample = binary(lineLetter[i]);
			auto d = predict(mlp, sample);
			std::cout << letters.substr(d * 2, 1);
		}
		++j;
		std::cout << std::endl;
	}
}

int PageSegmentation::predict(cv::Ptr<cv::ml::ANN_MLP> mlp, cv::Mat& sample) const
{
	Mat proc;
	cv::resize(sample, proc, Size(28, 28));
	sample = proc;
	/*cv::cvtColor(sample, proc, CV_BGR2GRAY);
	sample = proc;*/
	threshold(sample, sample, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);
	auto vec = convertMatToVec(sample);
	Mat res;

	// ReSharper disable once CppExpressionWithoutSideEffects
	mlp->predict(vec, res);
	float* row = res.ptr<float>(0);
	auto max = std::max_element(row, row + 33);
	int d = static_cast<int>(std::distance(row, max));
	return d;
}
