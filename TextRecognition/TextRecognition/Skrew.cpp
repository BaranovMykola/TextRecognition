#include "Skrew.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include <iostream>
#include <string>

#include "FileLoading.h"
#include "LetterDetection.h"
#include "LineSegmentation.h"
#include "Contants.h"
#include "ReleaseFunction.h"
#include <numeric>

using namespace cv;

namespace demo
{

	void SkrewRelease()
	{

		Mat img;
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
		}

		Mat preview;

		makePreview(img, preview);

		imshow("Original image", preview);

		waitKey();
		destroyAllWindows();

		auto thresh = letterHighligh(img);

		makePreview(thresh, preview);
		imshow("Binary img", preview);

		waitKey();
		destroyAllWindows();

		/**/

		int angle = 0;
		long long maxDev = 0;
		Mat resizedImage = preview.clone();
		Mat binary = preview.clone();

		namedWindow("Current Hist");
		namedWindow("Best Hist");
		namedWindow("Rotate");

		waitKey();

		int delay = 100;
		namedWindow("Speed");
		createTrackbar("Speed", "Speed", &delay, 1000);

		for (int i = -90; i <= 90; i += 5)
		{
			tryAngle(angle, i, resizedImage, maxDev);

			preview = rotate(binary, i);
			imshow("Rotate", preview);
			waitKey(delay);
		}

		for (int i = angle - 4; i <= angle + 4 && i != 0; i++)
		{
			tryAngle(angle, i, resizedImage, maxDev);

			preview = rotate(binary, i);
			imshow("Rotate", preview);
			waitKey(delay);
		}


		binary = rotate(binary, angle);
		imshow("Rotate", binary);
		destroyWindow("Current Hist");

		waitKey();
		destroyAllWindows();

		/**/
	}

	void tryAngle(int& angle, int newAngle, cv::Mat& resizedImage, long long& maxDev)
	{
		int min;
		int max;
		cv::Mat thresh = rotate(resizedImage, newAngle);
		auto freq = calculateProjectionHist(thresh, &min, &max);


		int aver = std::accumulate(freq.begin(), freq.end(), 0);
		aver /= static_cast<int>(freq.size());

		long long dev = static_cast<long long>(accumulate(freq.begin(), freq.end(), 0.0, [&](double acc, int elem)
		{
			return acc + pow((aver - elem), 2);
		}));

		auto hist = calculateGraphicHist(freq, max);
		imshow("Current Hist", hist);
		if (maxDev < dev)
		{
			imshow("Best Hist", hist);
			maxDev = dev;
			angle = newAngle;
		}
	}
}