#include <spatial_regularization.h>
#include <opencv2/ximgproc.hpp>
#include <vector>

void joint_bilateral_regularization(const cv::Mat &guide, cv::Mat &flow, int d , double sigma_color, double sigma_space) {
    if (flow.channels() != 2) {
        throw std::runtime_error("Flow must have 2 channels.");
    }
    // Split flow into its two channels
    std::vector<cv::Mat> flow_channels;
    cv::split(flow, flow_channels);

    cv::Mat guide_gray;
    if (guide.channels() == 3) {
        cv::cvtColor(guide, guide_gray, cv::COLOR_BGR2GRAY);
    } else {
        guide_gray = guide.clone();
    } 
    guide_gray.convertTo(guide_gray, CV_8UC1);

    // Apply joint bilateral filtering to each channel
    for (int i = 0; i < 2; ++i) {
        cv::ximgproc::jointBilateralFilter(guide_gray, flow_channels[i], flow_channels[i], d, sigma_color, sigma_space);
    }

    // merge channels back into flow
    cv::merge(flow_channels, flow);
}