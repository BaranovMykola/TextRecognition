#include <iostream>
#include <string>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include "Contants.h"
#include "LetterDetection.h"

void loadImg(cv::Mat& img, const std::string& fileName)
{
	img = cv::imread(TextSamplePathPrefix+fileName+".jpg");
	if (img.empty())
	{
		throw std::exception("Invalid name");
	}
}

int main(int argc, char* cargv[])
{
	std::string command;
	cv::Mat img;
	do
	{
		std::cout << ">> ";
		std::cin >> command;
		try
		{
			loadImg(img, command);
		}
		catch (const std::exception& e)
		{
			std::cout << "Error while opening: " << e.what() << std::endl;
			continue;
		}

		cv::namedWindow("Img", CV_WINDOW_KEEPRATIO);

		auto letterThresholded = letterHighligh(img);
		auto rects = encloseLetters(letterThresholded);
		extractLetters(rects, img);
		cv::imshow("Img", img);


		if (cv::waitKey() == 27)
		{
			return 0;
		}
		cv::destroyAllWindows();

	}
	while (command != "q");
	return 0;
}