#include "ReleaseFunction.h"
#include <opencv2\imgproc.hpp>

#include "Contants.h"

using namespace cv;

void demo::makePreview(cv::Mat& src, cv::Mat& dst)
{
	float sizeModifier = std::min({ src.cols / SkrewRestoringImageSize, src.rows / SkrewRestoringImageSize });
	sizeModifier = sizeModifier >= 1 ? sizeModifier : 1;
	cv::resize(src, dst, Size(src.cols / sizeModifier, src.rows / sizeModifier), 0, 0, INTER_NEAREST);
}
