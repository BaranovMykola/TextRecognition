#include "RectComparator.h"

bool RectComparator::operator()(const cv::Rect & lhs, const cv::Rect & rhs) const
{
		return lhs.y < rhs.y || lhs.y == rhs.y && lhs.x < rhs.x;
}
