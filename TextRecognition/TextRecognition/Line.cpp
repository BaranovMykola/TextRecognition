#include "Line.h"
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include <iostream>
#include <string>

#include "LineSegmentation.h"
#include "FileLoading.h"
#include "ReleaseFunction.h"

void demo::LineRelease()
{
	cv::Mat img;
	std::string action;
	std::cout << ">> ";
	std::cin >> action;
	try
	{
		loadImg(img, action);
	}
	catch (const std::exception& e)
	{
		std::cout << "Error while opening: " << e.what() << std::endl;
		return;
	}

	cv::Mat preview;
	makePreview(img, preview);
	cv::imshow("Origin", preview);
	cv::waitKey();
	cv::destroyAllWindows();

	auto thresh = letterHighligh(img);

	auto skew = findSkew(thresh);

	auto binary = rotate(thresh, skew);

	auto rectangles = encloseLetters(binary);

	for (auto i : rectangles)
	{
		binary(i) = 0;
	}



	makePreview(binary, preview);
	cv::imshow("Processed", preview);
	cv::waitKey();
	cv::destroyAllWindows();

	auto copy = binary.clone();
	int min;
	int max;
	std::vector<int> freq = calculateProjectionHist(copy, &min, &max);
	double average = std::accumulate(freq.begin(), freq.end(), 0) / (double)freq.size();
	double thresholdLevel = (average + min) / 2;
	threshold(freq, thresholdLevel, max);

	auto h = calculateGraphicHist(freq, max);

	for (int i = 0; i < freq.size(); i++)
	{
		if (freq[i] != max)
			continue;
		uchar* row = copy.ptr<uchar>(i);
		for (int j = 0; j < copy.cols; j++)
		{
			if (row[j] > 0)
				row[j] -= 127;
		}
	}

	makePreview(copy, preview);
	cv::imshow("Lines", preview);
	cv::waitKey();
 	cv::destroyAllWindows();
}
