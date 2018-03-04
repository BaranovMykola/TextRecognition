#pragma once
#include <vector>

namespace cv {
	class Mat;
}

namespace demo
{
	void LineRelease();
	void drawLines(std::vector<int> lines, cv::Mat& img);
}
