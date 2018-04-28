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
#include  <fstream>

#include "Contants.h"
#include "WordSegmentation.h"
#include "Deskew.h"
#include "LetterDetection.h"
#include <memory>

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
				labels.at<float>(r, k) = k == digit ? float(1) : float(0);
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
			int d = static_cast<int>(std::distance(row, max));
			if (d == digit)
			{
				++correct;
			}
			++total;
		}
		std::cout << "Total: " << total << "\tCorrect: " << correct << "\t(" << 100 * correct / (double)total << "%)" << endl;
		acc.push_back(static_cast<int>(100 * correct / static_cast<double>(total)));

		total = 0;
		correct = 0;
	}
	std::cout << "\tTotal accuracy:\t" << std::accumulate(acc.begin(), acc.end(), 0) / (double)classes << "%" << endl;
}

void printLetters(int count)
{
	setlocale(LC_CTYPE, "ukr");
	std::string letters = "ÀàÁáÂâÃãÄäÅåªºÆæÇçÈè²³¯¿ÉéÊêËëÌìÍíÎîÏïÐðÑñÒòÓóÔôÕõÖö×÷ØøÙùÜüÞþßÿ";
	for (auto i : letters)
	{
		for (int j = 0; j < count; j++)
		{
			cout << i << " ";
		}
		cout << endl;
		for (int j = 0; j < count; j++)
		{
			cout << i << " ";
		}
		cout << endl << endl;
	}
}

void cropDataset(cv::Mat& img, string path)
{
	auto thresh = letterHighligh(img);
	auto skew = findSkew(thresh);
	auto binary = mat::rotate(thresh, skew);
	threshold(binary, binary, 127, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	auto clone = binary.clone();
	Mat closed;
	morphologyEx(binary, closed, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(10, 20)));
	auto letters = encloseLetters(closed);

	int id = 0;
	for (auto& letter : letters)
	{
		imwrite(path + std::to_string(id) + ".jpg", binary(letter));
		++id;
	}
}

void prepareTrainData(std::string trainFile, std::string outputFile)
{
	ifstream train;
	ofstream data(outputFile);
	train.open(trainFile);

	string input;
	int label;
	int processSample = 0;
	while(!train.eof())
	{
		train >> input;
		train >> label;

		Mat sample = imread(input, CV_LOAD_IMAGE_GRAYSCALE);
		Mat resized;
		cv::resize(sample, resized, SAMPLE_SIZE);
		threshold(resized, resized, 127, 255, THRESH_BINARY);
		normalize(resized, resized, 0, 1, NORM_MINMAX);
		sample = resized;

		auto samples = dataAugmentation(sample);

		for (auto item : samples)
		{
			auto vec = convertMatToVec(item);
			writeVec<float>(data, vec);
			data << label;
			data << endl;
		}
	}
}

std::string convertIntToBitArray(int label, int size)
{
	string output;
	for (int i = 0; i < size; ++i)
	{
		output += std::to_string(i % 2);
		i /= 10;
	}
	return output;
}

std::vector<cv::Mat> dataAugmentation(cv::Mat& sample)
{
	return std::vector<cv::Mat>{ sample };
}

template <typename T>
void writeVec(std::ofstream& out, cv::Mat& vec)
{
	for (int i = 0; i < vec.rows; ++i)
	{
		T* row = vec.ptr<T>(i);
		for (int i = 0; i < vec.cols; ++i)
		{
			out << (int)row[i] << " ";
		}
	}
}
