#include <iostream>
#include "include/computeSymmetricFlow.h"
#include <opencv2/opencv.hpp>

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

int main()
{
    // Manually set your video path here
    std::string videoPath = "/path/to/video.mp4";

    std::vector<cv::Mat> frames;

    if (!loadVideoFrames(videoPath, frames)) {
        std::cerr << "Failed to load frames.\n";
        return 1;
    }

    std::cout << "Frames loaded: " << frames.size() << "\n";
    std::cout << "Resolution: "
    << frames[0].cols << " x "
    << frames[0].rows  << "\n";

    // Example: use frames for interpolation
    // cv::Mat f0 = frames[10];
    // cv::Mat f1 = frames[11];

    return 0;
}
bool loadVideoFrames(const std::string& videoPath, std::vector<cv::Mat>& frames)
{
    frames.clear();

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video: " << videoPath << "\n";
        return false;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;
        frames.push_back(frame.clone());
    }

    return !frames.empty();
}