#include "warpUtils.h"

// Bilinear sampling from RGB image at floating coordinates

bool sampleFrameBilinear(const cv::Mat& frame, float x, float y, cv::Vec3b& outPixel)
{
    if (x < 0.0f || y < 0.0f || x >= frame.cols - 1.0f || y >= frame.rows - 1.0f)
        return false;

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    float dx = x - x0;
    float dy = y - y0;

    const cv::Vec3b& p00 = frame.at<cv::Vec3b>(y0,     x0);
    const cv::Vec3b& p01 = frame.at<cv::Vec3b>(y0,     x0 + 1);
    const cv::Vec3b& p10 = frame.at<cv::Vec3b>(y0 + 1, x0);
    const cv::Vec3b& p11 = frame.at<cv::Vec3b>(y0 + 1, x0 + 1);

    cv::Vec3f r0 = p00 * (1.0f - dx) + p01 * dx;
    cv::Vec3f r1 = p10 * (1.0f - dx) + p11 * dx;
    cv::Vec3f r  = r0 * (1.0f - dy) + r1 * dy;

    outPixel = cv::Vec3b(
        cv::saturate_cast<uchar>(r[0]),
        cv::saturate_cast<uchar>(r[1]),
        cv::saturate_cast<uchar>(r[2])
    );
    return true;
}

