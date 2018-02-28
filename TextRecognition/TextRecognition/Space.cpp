#include "Space.h"
#include <opencv2/highgui.hpp>

#include <iostream>

#include "ReleaseFunction.h"
#include "FileLoading.h"
#include "LineSegmentation.h"
#include "WordSegmentation.h"
#include <opencv2/imgproc.hpp>

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

	auto binary = rotate(thresh, skew);
	
	makePreview(binary, preview);
	cv::imshow("Binary", preview);
	cv::waitKey();
	cv::destroyAllWindows();

	auto spaces = segmentWords(binary);

	for (auto space : spaces)
	{
		int line = space.first;
		for (auto pos : space.second)
		{
			cv::circle(binary, cv::Point(pos, line), 30, cv::Scalar::all(127), -1);
		}
	}

	makePreview(binary, preview);
	cv::imshow("Spaces", preview);
	cv::waitKey();
	cv::destroyAllWindows();
}