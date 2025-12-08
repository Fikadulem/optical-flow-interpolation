#include "spatialRegularization.h"
#include <vector>

// Apply edge-aware smoothing to optical flow using joint bilateral filter algorithm
void jointBilateralRegularization(const cv::Mat &guide, cv::Mat &flow,
                                  int d, double sigmaColor, double sigmaSpace) {
    CV_Assert(flow.type() == CV_32FC2);

    std::vector<cv::Mat> channels;
    cv::split(flow, channels);

    cv::Mat guideGray;
    if (guide.channels() == 3) {
        cv::cvtColor(guide, guideGray, cv::COLOR_BGR2GRAY);
    } else {
        guideGray = guide.clone();
    }
    guideGray.convertTo(guideGray, CV_32F);

    // Apply bilateral filter for edge-preserving smoothing
    for (int i = 0; i < 2; ++i) {
        cv::Mat tmp;
        cv::bilateralFilter(channels[i], tmp, d, sigmaColor, sigmaSpace);
        channels[i] = tmp;
    }

    cv::merge(channels, flow);
}