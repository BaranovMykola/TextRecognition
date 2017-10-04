#include <iostream>
#include <string>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <time.h>
#include <iostream>

#include "Contants.h"
#include "LetterDetection.h"
#include "TrainANN.h"

void loadImg(cv::Mat& img, const std::string& fileName)
{
	img = cv::imread(TextSamplePathPrefix+fileName+".jpg");
	if (img.empty())
	{
		throw std::exception("Invalid name");
	}
}

int main(int argc, char* cargv[])
{
	time_t currentTime = time(0);
	std::cout << "Started programm at\t" << asctime(localtime(&currentTime)) << std::endl;
	testANN(loadANN("letters_classifier"));
	currentTime = time(0);
	std::cout << "Finished programm at\t" << asctime(localtime(&currentTime)) << std::endl;
	std::cout << "Program stopped before exit... (Press any key to continue)" << std::endl;
	char ch;
	std::cin >> ch;
	return 0;
}