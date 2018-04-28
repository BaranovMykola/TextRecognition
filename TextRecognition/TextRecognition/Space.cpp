#include "Space.h"
#include <opencv2/highgui.hpp>

#include <iostream>

#include "ReleaseFunction.h"
#include "FileLoading.h"
#include "LineSegmentation.h"
#include "WordSegmentation.h"
#include <opencv2/imgproc.hpp>
#include "LetterDetection.h"
#include "Deskew.h"

void demo::spaceRelease()
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

	auto filled = mat::fillLetters(binary);
	
	/*makePreview(filled, preview);
	cv::imshow("Filled", preview);
	cv::waitKey();*/

	auto lines = detectLines(filled);
	auto sortedLetters = segmentAllLines(binary, lines);


	cv::Mat draw(binary.size(), CV_8UC1);
	draw = 255;

	for (int line = 0; line < lines.size(); ++line)
	{
		int s = 0;
		std::sort(sortedLetters[line].begin(), sortedLetters[line].end(), [](auto l, auto r) {return l.x < r.x; });
		auto spaces = checkSpaces(sortedLetters[line]);
		for (int i = 0; i < sortedLetters[line].size(); ++i)
		{
			if(s < spaces.size() && spaces[s] == i)
			{
				makePreview(draw, preview);
				cv::imshow("Word", preview);
				cv::waitKey(300);
				if (s < spaces.size()-1)
				{
					++s;
				}
			}
			binary(sortedLetters[line][i]).copyTo(draw(sortedLetters[line][i]));
		}
	}

	cv::destroyAllWindows();
}
