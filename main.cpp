#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace cv::ml;
using namespace std;

const int IMAGE_SIZE = 28;

vector<vector<unsigned char>> ReadImages(const string& fileName) {
    ifstream file(fileName, ios::binary);

    char magicNumber[4];
    char nImagesBytes[4];
    char nRowsBytes[4];
    char nColsBytes[4];

    file.read(magicNumber, 4);
    file.read(nImagesBytes, 4);
    file.read(nRowsBytes, 4);
    file.read(nColsBytes, 4);

    int nImages = (static_cast<unsigned char>(nImagesBytes[3]) << 0) |
        (static_cast<unsigned char>(nImagesBytes[2]) << 8) |
        (static_cast<unsigned char>(nImagesBytes[1]) << 16) |
        (static_cast<unsigned char>(nImagesBytes[0]) << 24);
    int nRows = (static_cast<unsigned>(nRowsBytes[3]) << 0) |
        (static_cast<unsigned>(nRowsBytes[2]) << 8) |
        (static_cast<unsigned>(nRowsBytes[1]) << 16) |
        (static_cast<unsigned>(nRowsBytes[0]) << 24);
    int nCols = (static_cast<unsigned>(nColsBytes[3]) << 0) |
        (static_cast<unsigned>(nColsBytes[2]) << 8) |
        (static_cast<unsigned>(nColsBytes[1]) << 16) |
        (static_cast<unsigned>(nColsBytes[0]) << 24);

    vector<vector<unsigned char>> result;

    for (int i = 0; i < nImages; i++) {
        vector<unsigned char> image(nRows * nCols);
        file.read((char*)(image.data()), nRows * nCols);
        result.push_back(image);
    }

    file.close();
    return result;
}

vector<vector<unsigned char>> ReadLabels(const string& fileName) {
    ifstream file(fileName, ios::binary);

    char magicNumber[4];
    char nImagesBytes[4];

    file.read(magicNumber, 4);
    file.read(nImagesBytes, 4);

    int nImages = (static_cast<unsigned char>(nImagesBytes[3]) << 0) |
        (static_cast<unsigned char>(nImagesBytes[2]) << 8) |
        (static_cast<unsigned char>(nImagesBytes[1]) << 16) |
        (static_cast<unsigned char>(nImagesBytes[0]) << 24);

    vector<vector<unsigned char>> result;

    for (int i = 0; i < nImages; i++) {
        vector<unsigned char> label(1);
        file.read((char*)(label.data()), 1);
        result.push_back(label);
    }

    file.close();
    return result;
}



int main() {
    vector<vector<unsigned char>> trainImages = ReadImages("/Users/edwardseanalexander/Documents/CUHK docs/year 2/CSC 3002/assignment 6/train-images.idx3-ubyte");
    vector<vector<unsigned char>> trainLabels = ReadLabels("/Users/edwardseanalexander/Documents/CUHK docs/year 2/CSC 3002/assignment 6/train-labels.idx1-ubyte");

    vector<vector<unsigned char>> testImages = ReadImages("/Users/edwardseanalexander/Documents/CUHK docs/year 2/CSC 3002/assignment 6/t10k-images.idx3-ubyte");
    vector<vector<unsigned char>> testLabels = ReadLabels("/Users/edwardseanalexander/Documents/CUHK docs/year 2/CSC 3002/assignment 6/t10k-labels.idx1-ubyte");


    Mat trainData(trainImages.size(), IMAGE_SIZE * IMAGE_SIZE, CV_32F);
    Mat trainLabelsMat(trainLabels.size(), 1, CV_32S);

    Mat testData(testImages.size(), IMAGE_SIZE * IMAGE_SIZE, CV_32F);
    Mat testLabelsMat(testLabels.size(), 1, CV_32S);

    for (size_t i = 0; i < trainImages.size(); i++) {
        for (size_t j = 0; j < trainImages[i].size(); j++) {
            trainData.at<float>(i, j) = static_cast<float>(trainImages[i][j]) / 255.0;
        }
        trainLabelsMat.at<int>(i, 0) = static_cast<int>(trainLabels[i][0]);
    }

    for (size_t i = 0; i < testImages.size(); i++) {
        for (size_t j = 0; j < testImages[i].size(); j++) {
            testData.at<float>(i, j) = static_cast<float>(testImages[i][j]) / 255.0;
        }
        testLabelsMat.at<int>(i, 0) = static_cast<int>(testLabels[i][0]);
    }

    Ptr<RTrees> randomForest = RTrees::create();
    randomForest->setMaxDepth(20);
    randomForest->setMinSampleCount(1);
    randomForest->setUseSurrogates(false);
    randomForest->setMaxCategories(15);
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0.01));
    randomForest->train(trainData, ROW_SAMPLE, trainLabelsMat);

    Mat predictionsRF;
    randomForest->predict(testData, predictionsRF);

    cout << "Random Forest Predictions:" << endl;
    for (int i = 0; i < predictionsRF.rows; i++) {
        cout << "Real digit: " << testLabelsMat.at<int>(i, 0) << " Recognized digit (Random Forest): " << static_cast<int>(predictionsRF.at<float>(i, 0)) << endl;
    }

    int correctRF = 0;
    for (int i = 0; i < predictionsRF.rows; i++) {
        if (static_cast<int>(predictionsRF.at<float>(i, 0)) == testLabelsMat.at<int>(i, 0)) {
            correctRF++;
        }
    }

    double accuracyRF = static_cast<double>(correctRF) / predictionsRF.rows;
    cout << "Accuracy (Random Forest): " << accuracyRF * 100 << "%" << endl;

    Ptr<KNearest> knn = KNearest::create();
    knn->setIsClassifier(true);
    knn->setAlgorithmType(KNearest::BRUTE_FORCE);
    knn->setDefaultK(7);
    knn->setEmax(10);
    knn->train(trainData, ROW_SAMPLE, trainLabelsMat);

    Mat predictionsKNN;
    knn->findNearest(testData, 3, predictionsKNN);

    cout << "KNN Predictions:" << endl;
    for (int i = 0; i < predictionsKNN.rows; i++) {
        cout << "Real digit: " << testLabelsMat.at<int>(i, 0) << " Recognized digit (KNN): " << static_cast<int>(predictionsKNN.at<float>(i, 0)) << endl;
    }

    int correctKNN = 0;
    for (int i = 0; i < predictionsKNN.rows; i++) {
        if (static_cast<int>(predictionsKNN.at<float>(i, 0)) == testLabelsMat.at<int>(i, 0)) {
            correctKNN++;
        }
    }

    double accuracyKNN = static_cast<double>(correctKNN) / predictionsKNN.rows;
    cout << "Accuracy (KNN): " << accuracyKNN * 100 << "%" << endl;

    return 0;
}