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

#pragma once
#include <opencv2\core.hpp>
#include <opencv2\ml.hpp>

/**
 * \brief Convert 2D matrix to 1D vector
 * \param mat Arbitary matrix
 * \return Return vector from matrix
 */
cv::Mat convertMatToVec(const cv::Mat & mat);

/**
 * \brief Create train data from dataset
 * \return Returns train data
 */
cv::Ptr<cv::ml::TrainData> createTrainData();

/**
 * \brief Trains ANN and save to file when trained
 * \param saveTo File to save ANN
 */
void trainANN(std::string saveTo);

/**
 * \brief Loadd ANN from file
 * \param file File to load ANN from
 * \return Returns ANN loaded from file
 */
cv::Ptr<cv::ml::ANN_MLP> loadANN(std::string file);

/**
 * \brief Test ANN. Get result as standart output
 * \param mlp ANN to test
 */
void testANN(cv::Ptr<cv::ml::ANN_MLP> mlp);

/**
 * \brief Prints all ukranians letters
 * \param count Count of each letters
 */
void printLetters(int count);