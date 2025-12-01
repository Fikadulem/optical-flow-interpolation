#ifndef SPATIAL_REGULARIZATION_H
#define SPATIAL_REGULARIZATION_H

#include <opencv2/opencv.hpp>

// Apply joint bilateral filetering fro optical flow spatial regularization
void joint_bilateral_regularization(const cv::Mat &guide, cv::Mat &flow, 
                                    int d = 9, double sigma_color = 25.0, double sigma_space = 25.0);

#endif // SPATIAL_REGULARIZATION_H