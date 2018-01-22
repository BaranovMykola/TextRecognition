#pragma once
#include <opencv2\core.hpp>

struct RectComparator
{
	bool operator() (const cv::Rect& lhs, const cv::Rect& rhs) const;
};

