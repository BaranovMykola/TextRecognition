/*
*  This file stored contstancs such as paths and sizes
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
#include <string>
#include <opencv2/core.hpp>

/**
 * \brief Path prefix to 'samples' folder
 */
const std::string SamplePathPrefix = "../samples/";

/**
 * \brief Path prefix to 'Classifier' fodler
 */
const std::string ClassifierPrefix = "../Classifier/";

/**
 * \brief Path to original images
 */
const std::string TextSamplePathPrefix = SamplePathPrefix + "letters/original/";

/**
 * \brief Path to test images
 */
const std::string TestSamplePathPrefix = SamplePathPrefix + "letters/original/new_dataset/";

/**
 * \brief Path to dataset source images
 */
const std::string TextDatasetPathPrefix = SamplePathPrefix + "dataset/source/";

/**
 * \brief Max resolution of images to skrew detection
 */
const float SkrewRestoringImageSize = 500;

/**
 * \brief Size of morphology kernel size that uses for binary image preprocessing before line detection
 */
const int HORIZONTAL_LINE_MORPHOLOGY_KERNEL_SIZE = 40;

const int HISTOGRAM_BLUR_KERNEL_SIZE = 20;

const cv::Size SAMPLE_SIZE = cv::Size(28, 28);

const int AFFINE_SAMPLES = 10;

const int NOISE_SAMPLES = 3;

const double AFFINE_MAX_ELEMENT = 0.1;

const int AFFINE_ADDITIONAL_ELEMENTS = 3;

const int NOISE_COUNT = 10;