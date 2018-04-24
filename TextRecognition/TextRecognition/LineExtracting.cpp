#include "LineExtracting.h"
#include <ostream>
#include <iostream>
#include "FileLoading.h"
#include "LineSegmentation.h"
#include <opencv2/highgui.hpp>
#include "ReleaseFunction.h"
#include "Line.h"
#include "LetterDetection.h"

void demo::extractLines()
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

	int shift = averLetterHight(binary) / 3 - 5 + 4;

	auto backup = binary.clone();

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


	cv::Mat draw(binary.size(), CV_8UC1);
	draw = 255;

	binary = closeCharacters(binary);

	auto allLetters = encloseLetters(binary);
	cv::namedWindow("Result", CV_WINDOW_FREERATIO);
	auto allSortedLetters = segmentAllLines(binary, lines);

	for (auto i : allSortedLetters)
	{
		std::sort(i.begin(), i.end(), [](cv::Rect l, cv::Rect r) {return l.x < r.x; });
		for (auto ch : i)
		{
			ch.y -= shift;
			backup(ch).copyTo(draw(ch));
			cv::imshow("Result", draw);
			cv::waitKey(1);
		}
	}

}
