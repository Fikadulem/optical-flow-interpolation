#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include "computeSymmetricFlow.h"
#include "spatialRegularization.h"
#include "warpUtils.h"
#include "occlusionHandling.h"
#include <iostream>
using namespace cv;

// Compute Mean Absolute Interpolation Error
double computeMAIE(const cv::Mat& pred, const cv::Mat& gt) {
    CV_Assert(pred.size() == gt.size());
    CV_Assert(pred.type() == gt.type());

    Mat diff;
    absdiff(pred, gt, diff);
    Scalar mae = mean(diff);

    double maie = 0.0;
    for (int i = 0; i<diff.channels(); ++i) {
        maie += mae[i];
    }
    return maie / diff.channels();

    }


// main function
int main(int argc, char** argv) {

    if (argc < 4) {
        std::cerr << "Usage: " <<argv[0] << " <frame10> <frame11> <frame10i1>\n";
        return -1;
    }

    Mat frame10 = imread(argv[1]);
    Mat frame11 = imread(argv[2]);
    Mat frame10i11 = imread(argv[3]);
    if (frame10.empty() || frame11.empty() || frame10i11.empty()) {
        std::cerr << "Error reading input images.\n";
        return -1;
    }

    if (frame10.size() != frame11.size()) {
        std::cerr << "Input frames must have the same size.\n";
        return -1;
    }

    
    // Before spatial regulazirization 
    Mat vsRaw;
    if (!computeSymmetricFlowFarneback(frame10, frame11, vsRaw)) {
        std::cerr << "Failed to compute raw symmetric flow.\n";
        return -1;
    }
    Mat interpRaw = interpolateSymmetric(frame10, frame11, vsRaw);
    double maieRaw = computeMAIE(interpRaw, frame10i11);
    Mat interpolated = interpolateSymmetric(frame10, frame11, vsRaw);
    
    // Compute Mean Absolute Interpolation Error
    double maie = computeMAIE(interpolated, frame10i11);
    std::cout << "Mean Absolute Interpolation Error before Spatial Regularization= " << maie << std::endl;

    // After SpatialRegularization
    Mat vsReg;
    Mat g0, g1;
    if (frame10.channels() ==3) cvtColor(frame10, g0, COLOR_BGR2GRAY); else g0 = frame10.clone();
    if (frame11.channels() ==3) cvtColor(frame11, g1, COLOR_BGR2GRAY); else g1 = frame11.clone();
    computeSymmetricFlowFarneback(frame10, frame11, vsReg);
    jointBilateralRegularization(g0, vsReg, 5, 20.0, 20.0);
    // Compute occlusion mask (forward-backward consistency)
    Mat occMask = computeOcclusionMaskFarneback(frame10, frame11, 1.0f);
    // Interpolate with occlusion awareness
    Mat interpReg = interpolateSymmetricWithOcclusion(frame10, frame11, vsReg, occMask);
    
    // Compute Mean Absolute Interpolation Error
    double maieReg = computeMAIE(interpReg, frame10i11);
    std::cout << "Mean Absolute Interpolation Error after Spatial Regularization = " << maieReg << std::endl;
    interpolated = interpReg;
    
    // Display and save results  
    imshow("Frame 10", frame10);
    imwrite("frame10.jpg", frame10);
    imshow("Frame 11", frame11);
    imwrite("frame11.jpg", frame11);
    imshow("Interpolated Midpoint", interpolated);
    imwrite("interpolated_midpoint.jpg", interpolated);
    waitKey(0);

    return 0;
}