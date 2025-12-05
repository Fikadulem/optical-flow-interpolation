#include <opencv2/opencv.hpp>
#include "computeSymmetricFlow.h"
#include "warpUtils.h"
#include <iostream>

// main fnction

int main(int argc, char** argv) {
    std::string videoPath = "video.mp4";
    if (argc >= 2) videoPath = argv[1];

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video file: " << videoPath << "\n";
        return -1;
    }

    cv::Mat frame0, frame1;
    cap >> frame0;
    cap >> frame1;

    if (frame0.empty() || frame1.empty()) {
        std::cerr << "Not enough frames to interpolate.\n";
        return -1;
    }

    cv::Mat vs;
    if (!computeSymmetricFlowTVL1(frame0, frame1, vs)) {
        std::cerr << "Failed to compute symmetric flow.\n";
        return -1;
    }

    cv::Mat interpolated = interpolateSymmetric(frame0, frame1, vs);

    cv::imshow("Frame 0", frame0);
    cv::imshow("Frame 1", frame1);
    cv::imshow("Interpolated Midpoint", interpolated);
    cv::waitKey(0);

    return 0;
}