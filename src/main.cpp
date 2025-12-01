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

int main() {
    std::cout << "Program started!" << std::endl;
    return 0;
}
