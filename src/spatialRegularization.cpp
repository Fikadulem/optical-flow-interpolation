#include "spatialRegularization.h"
#include <opencv2/ximgproc.hpp>

/**
* Precondtions:
* - flow must be 2 channnl
* - Giude must be single channe
* - Must be positive integer
* - sigmaColor and sigmaSPace must be positve number
* - OpenCV `ximgproc`
* Postconditions
* - flow must be in place
* - Smoothed flow
* - size and type of flow unchanged
*/

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

    for (int i = 0; i < 2; ++i) {
        cv::ximgproc::jointBilateralFilter(
            guideGray, channels[i], channels[i],
            d, sigmaColor, sigmaSpace
        );
    }

    cv::merge(channels, flow);
}