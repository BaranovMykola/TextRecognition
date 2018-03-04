/*
*  This file contains functions that used to train or test ANN
*  Copyright (C) 2018 Mykola Baranov
*
*  This program is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "TrainANN.h"
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>
#include <numeric>

#include "Contants.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

cv::Mat convertMatToVec(const cv::Mat& mat)
{
	Mat f;
	mat.convertTo(f, CV_32F);
	Mat vec = Mat::zeros(1, 784, CV_32FC1);
	for (int i = 0; i < 784; i++)
	{
		vec.at<float>(0, i) = (float)f.at<float>(i / 28, i % 28);
	}
	return vec;
}

Ptr<TrainData> createTrainData()
{
	std::string path = "../samples/dataset/train_data/";
	int samplesCount = 15490+1007;
	Mat trainData(samplesCount, 784, CV_32FC1);
	Mat labels(samplesCount, 33, CV_32FC1);
	int r = 0;
	cout << "Loading train data..." << endl;
	for (int digit = 0; digit < 33; digit++)
	{
		cout << "Processing " << digit << " class" << endl;
		for (int i = 0; i <= 2000; i++)
		{
			std::string file = path + std::to_string(digit) + "/" + std::to_string(i) + ".jpg";
			Mat sample = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
			threshold(sample, sample, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);
			if (sample.empty())
			{
				continue;
			}
			auto vec = convertMatToVec(sample);
			vec.row(0).copyTo(trainData.row(r));

			for (int k = 0; k < 33; k++)
			{
				labels.at<float>(r, k) = k == digit ? 1 : 0;
			}

			++r;
		}
	}
	Ptr<TrainData> trainingData;
	trainingData = TrainData::create(trainData, SampleTypes::ROW_SAMPLE, labels);
	cout << "Train data loaded" << endl;
	return trainingData;
}

void trainANN(std::string saveTo)
{
	auto t = createTrainData();
	Ptr<ANN_MLP> mlp;
	mlp = ANN_MLP::create();


	Mat layersSize = Mat(4, 1, CV_16U);
	layersSize.row(0) = Scalar(784);
	layersSize.row(1) = Scalar(28);
	layersSize.row(2) = Scalar(28);
	layersSize.row(3) = Scalar(33);
	mlp->setLayerSizes(layersSize);

	mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM, 0, 1);

	TermCriteria termCrit = TermCriteria(
		TermCriteria::Type::COUNT + TermCriteria::Type::EPS,
		10000,
		0.1
	);
	mlp->setTermCriteria(termCrit);

	mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP, 0.001);

	cout << "Starting training ANN..." << endl;
	mlp->train(t
			   /*, ANN_MLP::TrainFlags::UPDATE_WEIGHTS
			   + ANN_MLP::TrainFlags::NO_INPUT_SCALE
			   + ANN_MLP::TrainFlags::NO_OUTPUT_SCALE*/
	);
	cout << "ANN trained" << endl;
	cout << "Saving ANN..." << endl;
	mlp->save(ClassifierPrefix+saveTo);
	cout << "ANN saved" << endl;
}

Ptr<cv::ml::ANN_MLP> loadANN(std::string file)
{
	cout << "Loading ANN..." << endl;
	return ANN_MLP::load(file);
}

void testANN(cv::Ptr<cv::ml::ANN_MLP> mlp)
{
	int total = 0;
	int correct = 0;
	std::vector<int> acc;
	int classes = 33;
	int allClasses = 33;
	for (int digit = 0; digit < allClasses; digit++)
	{
		std::cout << "Processing " << digit << " class of samples... ";
		for (int i = 0; i < 2000; i++)
		{
			std::string path = "../samples/dataset/test_data/";
			std::string file = path + std::to_string(digit) + "/" + std::to_string(i) + ".jpg";
			auto sample = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
			threshold(sample, sample, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);
			if (sample.empty())
			{
				//cout << "File " << file << " not found" << endl;
				continue;
			}
			auto vec = convertMatToVec(sample);
			Mat res;
			
			// ReSharper disable once CppExpressionWithoutSideEffects
			mlp->predict(vec, res);
			float* row = res.ptr<float>(0);
			auto max = std::max_element(row, row + allClasses);
			int d = std::distance(row, max);
			if (d == digit)
			{
				++correct;
			}
			++total;
		}
		std::cout << "Total: " << total << "\tCorrect: " << correct << "\t(" << 100 * correct / (double)total << "%)" << endl;
		acc.push_back(100 * correct / (double)total);

		total = 0;
		correct = 0;
	}
	std::cout << "\tTotal accuracy:\t" << std::accumulate(acc.begin(), acc.end(), 0) / (double)classes << "%" << endl;
}

void printLetters(int count)
{
	setlocale(LC_CTYPE, "ukr");
	std::string letters = "ÀàÁáÂâÃã¥´ÄäÅåªºÆæÇçÈè²³¯¿ÉéÊêËëÌìÍíÎîÏïÐðÑñÒòÓóÔôÕõÖö×÷ØøÙùÜüÞþßÿ";
	for (auto i : letters)
	{
		for (int j = 0; j < count; j++)
		{
			cout << i << " ";
		}
		cout << endl << endl;
	}
}
