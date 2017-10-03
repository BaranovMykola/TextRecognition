#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\ml.hpp>
#include <string>
#include <vector>
#include <numeric>

using namespace cv;
using namespace cv::ml;
using namespace std;

#include "TrainANN.h"

int main(int argc, char* argv[])
{
	printLetters();
	system("pause");
	return 0;
}