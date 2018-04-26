#include "Line.h"
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include <iostream>
#include <string>

#include "LineSegmentation.h"
#include "FileLoading.h"
#include "ReleaseFunction.h"
#include "LetterDetection.h"
#include <numeric>
#include <opencv2/imgproc.hpp>
#include "WordSegmentation.h"
#include "Deskew.h"

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

	auto binary = mat::rotate(thresh, skew);

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
	
	auto lines = detectLines(binary);

	drawLines(lines, copy);

	makePreview(copy, preview);
	cv::imshow("Lines", preview);
	cv::waitKey();
 	cv::destroyAllWindows();

	lines = clearMultipleLines(lines, binary);

	drawLines(lines, binary);

	makePreview(binary, preview);
	cv::imshow("Lines", preview);
	cv::waitKey();
	cv::destroyAllWindows();
}

void demo::drawLines(std::vector<int> lines, cv::Mat& img)
{
	for (auto line : lines)
	{
		cv::line(img, cv::Point(0, line), cv::Point(img.cols, line), cv::Scalar::all(127), 3);
	}
}

