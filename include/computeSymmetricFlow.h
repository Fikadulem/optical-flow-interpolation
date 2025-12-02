#ifndef COMPUTE_SYMMETRIC_FLOW_H
#define COMPUTE_SYMMETRIC_FLOW_H

#include <opencv2/opencv.hpp>

void computeSymmetricFlow(const cv::Mat &I0, const cv::Mat &I1,
                          cv::Mat &flowFwdHalf, cv::Mat &flowBwdHalf);

#endif // COMPUTE_SYMMETRIC_FLOW_H