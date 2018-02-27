#include "FileLoading.h"

#include <opencv2\highgui.hpp>

#include "Contants.h"

void loadImg(cv::Mat& img, const std::string& fileName)
{
	img = cv::imread(TestSamplePathPrefix + fileName + ".jpg");
	if (img.empty())
	{
		throw std::exception("Invalid name");
	}
}