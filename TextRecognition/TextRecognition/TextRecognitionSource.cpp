#include <iostream>
#include <string>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <time.h>
#include <iostream>

#include "Contants.h"
#include "LetterDetection.h"
#include "TrainANN.h"
#include "LineSegmentation.h"

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
	time_t currentTime = time(0);
	std::string action;
	std::cout << ">> ";
	std::cin >> action;
	std::cout << "Started programm at\t" << asctime(localtime(&currentTime)) << std::endl;
	if (action == "train")
	{
		trainANN("28x28_MLP_H0");
	}
	else if (action == "test")
	{
		testANN(loadANN(ClassifierPrefix+"28x28_MLP_H0"));
	}
	else if (action == "print")
	{
		int count;
		std::cout << "count >> ";
		std::cin >> count;
		printLetters(count);
	}
	else if (action == "crop")
	{
		cv::Mat img;
		do
		{
			std::cout << ">> ";
			std::cin >> action;
			try
			{
				loadImg(img, action);
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
		while (action != "q");
	}
	else if (action == "line")
	{
		do
		{
			cv::Mat img;
			std::cout << ">> ";
			std::cin >> action;
			try
			{
				loadImg(img, action);
			}
			catch (const std::exception& e)
			{
				std::cout << "Error while opening: " << e.what() << std::endl;
				continue;
			}

			auto thresh = letterHighligh(img);
			int min;
			int max;
			auto freq = calculateProjectionHist(thresh, &min, &max);
			auto hist = calculateGraphicHist(freq, max);

		}
		while (true);
	}

	currentTime = time(0);
	std::cout << "Finished programm at\t" << asctime(localtime(&currentTime)) << std::endl;
	std::cout << "Program stopped before exit... (Press any key to continue)" << std::endl;
	char ch;
	std::cin >> ch;
	return 0;
}