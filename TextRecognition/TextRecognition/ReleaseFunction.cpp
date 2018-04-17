#include "ReleaseFunction.h"
#include <opencv2\imgproc.hpp>

#include "Contants.h"

using namespace cv;

void demo::makePreview(cv::Mat& src, cv::Mat& dst)
{
	float sizeModifier = std::min({ src.cols / SkrewRestoringImageSize, src.rows / SkrewRestoringImageSize });
	sizeModifier = sizeModifier >= 1 ? sizeModifier : 1;
	cv::resize(src, dst, Size(static_cast<int>(src.cols / sizeModifier), static_cast<int>(src.rows / sizeModifier)), 0, 0, INTER_AREA);
}
