#ifndef WARP_UTILS_H
#define WARP_UTILS_H
#include <opencv2/opencv.hpp>

// Bilinear sampling from RGB image at floating ccordinates
bool sampleFrameBilinear(const cv::Mat& frame, float x, float y, cv::Vec3b& outPixel);

// Interpolate mid-point frame using symmetric flow
cv::Mat interpolateSymmetric(const cv::Mat& I0, const cv::Mat& I1, const cv::Mat& vs);

#endif // WARP_UTILS_H