// Declaration for computeSymmetricFlowTVL1 implemented in src/frame_interpolation.cpp
#ifndef COMPUTE_SYMMETRIC_FLOW_H
#define COMPUTE_SYMMETRIC_FLOW_H

#include <opencv2/core.hpp>

// Compute symmetric optical flow between two frames using TV-L1.
// I0, I1 : input images (CV_8UC1 or CV_8UC3)
// vs     : output symmetric flow field (CV_32FC2)
// returns: true on success
bool computeSymmetricFlowFarneback(const cv::Mat& I0, const cv::Mat& I1, cv::Mat& vs);

cv::Mat interpolateSymmetric(
    const cv::Mat& I0,
    const cv::Mat& I1,
    const cv::Mat& vs)
;
#endif // COMPUTE_SYMMETRIC_FLOW_H
