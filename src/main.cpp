#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/computeSymmetricFlow.h"

cv::Mat warpFrame(const cv::Mat &frame, const cv::Mat &flow) {
    cv::Mat map_x(frame.size(), CV_32FC1);
    cv::Mat map_y(frame.size(), CV_32FC1);

    for (int y = 0; y < frame.rows; ++y) {
        for (int x = 0; x < frame.rows; ++x) {
            cv::Point2f f = flow.at<cv::Point2f>(y, x);
            map_x.at<float>(y, x) = x + f.x;
            map_y.at<float>(y, x) = y + f.y;
        }

    }
    cv::Mat warped;
    cv::remap(frame, warped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return warped;

}
bool loadVideoFrames(const std::string& videoPath, std::vector<cv::Mat>& frames)
{
    frames.clear();

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video file: " << videoPath << "\n";
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;
        frames.push_back(frame.clone());
    }

    return !frames.empty();
}
int main()
{
    cv::Mat frame0 = cv::imread("C:\\Users\\olade\\Downloads\\eval-color-twoframes\\eval-data\\Basketball\\frame10.png");
    cv::Mat frame1 = cv::imread("C:\\Users\\olade\\Downloads\\eval-color-twoframes\\eval-data\\Basketball\\frame11.png");
    cv:: Mat vs;
    if (!computeSymmetricFlowFarneback(frame0, frame1, vs)) {
        std::cerr << "Error: could not compute symmetric flow.\n";
        return -1;
    }
    std::cout << "Frames loaded: 2\n";
    std::cout << "Resolution: "
    << frame0.cols << " x "
    << frame0.rows  << "\n";

    cv::Mat midFrame = interpolateSymmetric(frame0, frame1, vs);
    cv::imwrite("C:\\Users\\olade\\Downloads\\midframe.png", midFrame);
    std::cout << "Midpoint frame saved as midframe.png\n";
    // Example: use frames for interpolation
    // cv::Mat f0 = frames[10];
    // cv::Mat f1 = frames[11];

    return 0;
}
