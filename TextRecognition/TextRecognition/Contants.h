#pragma once
#include <string>

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
const std::string TestSamplePathPrefix = SamplePathPrefix + "letters/test/";

/**
 * \brief Path to dataset source images
 */
const std::string TextDatasetPathPrefix = SamplePathPrefix + "dataset/source/";

/**
 * \brief Max resolution of images to skrew detection
 */
const float SkrewRestoringImageSize = 500;
