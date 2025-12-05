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

cv::Mat interpolateSymmetric(const cv::Mat& I0, const cv::Mat& I1, const cv::Mat& vs)
{
    CV_Assert(I0.size() == I1.size());
    CV_Assert(I0.type() == CV_8UC3 && I1.type() == CV_8UC3);
    CV_Assert(vs.size() == I0.size() && vs.type() == CV_32FC2);

    cv::Mat I_mid(I0.size(), CV_8UC3);
    const int W = I0.cols;
    const int H = I0.rows;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            cv::Point2f v = vs.at<cv::Point2f>(y, x);

            float x0 = x - v.x;
            float y0 = y - v.y;
            float x1 = x + v.x;
            float y1 = y + v.y;

            cv::Vec3b c0, c1;
            bool valid0 = sampleFrameBilinear(I0, x0, y0, c0);
            bool valid1 = sampleFrameBilinear(I1, x1, y1, c1);

            cv::Vec3b out;
            if (valid0 && valid1) {
                out[0] = static_cast<uchar>((c0[0] + c1[0]) * 0.5f);
                out[1] = static_cast<uchar>((c0[1] + c1[1]) * 0.5f);
                out[2] = static_cast<uchar>((c0[2] + c1[2]) * 0.5f);
            } else if (valid0) {
                out = c0;
            } else if (valid1) {
                out = c1;
            } else {
                const cv::Vec3b& p0 = I0.at<cv::Vec3b>(y, x);
                const cv::Vec3b& p1 = I1.at<cv::Vec3b>(y, x);
                out[0] = static_cast<uchar>((p0[0] + p1[0]) * 0.5f);
                out[1] = static_cast<uchar>((p0[1] + p1[1]) * 0.5f);
                out[2] = static_cast<uchar>((p0[2] + p1[2]) * 0.5f);
            }
            I_mid.at<cv::Vec3b>(y, x) = out;
        }
    }
    return I_mid;
}