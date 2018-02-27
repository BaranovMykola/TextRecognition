#pragma once
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>

#include <string>

#include "LetterDetection.h"
#include "LineSegmentation.h"
#include "WordSegmentation.h"
#include "Contants.h"

std::string recognize(cv::Mat& img);
