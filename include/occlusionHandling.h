#ifndef OCCLUSION_HANDLING_H
#define OCCLUSION_HANDLING_H
#include <opencv2/opencv.hpp>

// PERFORM OCCLUSION HANDLING DURING INTERPOLATION
cv::Mat computeOcclusionMask(const cv::Mat &flowFwd, const cv::Mat &flowBwd, float threshold = 1.0f);

// Convenience: compute forward/backward Farneback flows internally and return consistency mask
cv::Mat computeOcclusionMaskFarneback(const cv::Mat& I0, const cv::Mat& I1, float threshold = 1.0f);

#endif // OCCLUSION_HANDLING_H