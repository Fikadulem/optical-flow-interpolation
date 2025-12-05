#ifndef SPATIAL_REGULARIZATION_H
#define SPATIAL_REGULARIZATION_H
#include <opencv2/opencv.hpp>

// Apply edge-aware smoothing to optical flow using joint bilateral filter algorithm
void jointBilateralRegularization(const cv::Mat &guide, cv::Mat &flow,
                                  int d = 5, double sigmaColor = 20.0, double sigmaSpace = 20.0);

#endif // SPATIAL_REGULARIZATION_H