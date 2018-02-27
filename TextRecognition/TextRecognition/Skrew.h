#pragma once

#include <opencv2\core.hpp>

namespace demo
{

	void SkrewRelease();

	void makePreview(cv::Mat & src, cv::Mat & dst);

	void tryAngle(int & angle, int newAngle, cv::Mat & resizedImage, long long & maxDev);

}
