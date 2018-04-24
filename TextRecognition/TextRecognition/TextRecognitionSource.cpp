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
#include "WordSegmentation.h"
#include "FileLoading.h" 
#include "Skrew.h"
#include "Line.h"
#include "Space.h"
#include <opencv2/imgproc.hpp>
#include "LineExtracting.h"

int main(int argc, char* cargv[])
{
	time_t currentTime = time(0);
	std::string action;
	std::cout << ">> ";
	std::cin >> action;

	tm tms;
	char timeStamp[100];
	localtime_s(&tms, &currentTime);
	asctime_s(timeStamp, sizeof timeStamp, &tms);

	std::cout << "Started programm at\t" << timeStamp << std::endl;
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
		
			auto skew = findSkew(thresh);

			auto binary = rotate(thresh, skew);

			cv::threshold(binary, binary, 127, 255, CV_THRESH_BINARY);

			auto lines = detectLines(binary.clone());

			lines = clearMultipleLines(lines, binary);

			//segmentExactLine(lines[0], binary);

			segmentAllLines(binary, lines);
		}
		while (true);
	}
	else if (action == "char")
	{
		std::cout << ">> ";
		std::cin >> action;
		cv::Mat img;
		loadImg(img, action);

		img = letterHighligh(img);

		img = rotate(img, findSkew(img));

		img = closeCharacters(img);

		encloseLetters(img);
	}
	else if (action == "word")
	{
		std::cout << ">> ";
		std::cin >> action;
		cv::Mat img;
		loadImg(img, action);
	}
	else if (action == "_skew")
	{
		demo::SkrewRelease(); // 1,2,3,4,6,7
	}
	else if (action == "_line")
	{
		demo::LineRelease(); // 1,2,3,4,6,7
	}
	else if (action == "_space")
	{
		demo::spaceRelease(); // 1,2,3,4,6,7
	}
	else if(action == "_segmentLine")
	{
		demo::extractLines();
	}

	currentTime = time(0);
	localtime_s(&tms, &currentTime);
	asctime_s(timeStamp, sizeof timeStamp, &tms);
	std::cout << "Finished programm at\t" << timeStamp << std::endl;
	std::cout << "Program stopped before exit... (Press any key to continue)" << std::endl;
	char ch;
	std::cin >> ch;
	return 0;
}