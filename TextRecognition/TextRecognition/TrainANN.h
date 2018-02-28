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