#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include "computeSymmetricFlow.h"
#include "warpUtils.h"
#include <iostream>

// Compute Mean Absolute Interpolation Error
double computeMAIE(const cv::Mat& pred, const cv::Mat& gt) {
    CV_Assert(pred.size() == gt.size());
    CV_Assert(pred.type() == gt.type());

    cv::Mat diff;
    cv::absdiff(pred, gt, diff);
    cv::Scalar mae = cv::mean(diff);

    double maie = 0.0;
    for (int i = 0; i<diff.channels(); ++i) {
        maie += mae[i];
    }
    return maie / diff.channels();

    }


// main fnction
int main(int argc, char** argv) {

    if (argc < 4) {
        std::cerr << "Usage: " <<argv[0] << " <frame10> <frame11> <frame10i1>\n";
        return -1;
    }

    cv::Mat frame10 = cv::imread(argv[1]);
    cv::Mat frame11 = cv::imread(argv[2]);
    cv::Mat frame10i11 = cv::imread(argv[3]);
    if (frame10.empty() || frame11.empty() || frame10i11.empty()) {
        std::cerr << "Error reading input images.\n";
        return -1;
    }

    if (frame10.size() != frame11.size()) {
        std::cerr << "Input frames must have the same size.\n";
    }

    // Compute symmetric flow
    cv::Mat vs;
    if (!computeSymmetricFlowTVL1(frame10, frame11, vs)) {
        std::cerr << "Failed to compute symmetric flow.\n";
        return -1;
    }


    cv::Mat interpolated = interpolateSymmetric(frame10, frame11, vs);

    // Compute Mean Absolute Interpolation Error
    double maie = computeMAIE(interpolated, frame10i11);
    std::cout << "Mean Absolute Interpolation Error = " << maie << std::endl;

    cv::imshow("Frame 10", frame10);
    cv::imwrite("frame10.jpg", frame10);
    cv::imshow("Frame 11", frame11);
    cv::imwrite("frame11.jpg", frame11);
    cv::imshow("Interpolated Midpoint", interpolated);
    cv::imwrite("interpolated_midpoint.jpg", interpolated);
    cv::waitKey(0);

    return 0;
}