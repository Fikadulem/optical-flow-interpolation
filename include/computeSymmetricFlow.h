#ifndef COMPUTE_SYMMETRIC_FLOW_H
#define COMPUTE_SYMMETRIC_FLOW_H
#include <opencv2/opencv.hpp>

// Computes symmetric flow to midpoint using TVL1
bool computeSymmetricFlowTVL1(const cv::Mat& I0, const cv::Mat& I1, cv::Mat& vs);

#endif // COMPUTE_SYMMETRIC_FLOW_H