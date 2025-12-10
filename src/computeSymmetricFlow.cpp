#include "computeSymmetricFlow.h"
#include "spatialRegularization.h"
#include <opencv2/optflow.hpp>
#include <iostream>

using namespace cv;
// Compute symmetric flow v_s between I0 and I1 using TV-L1.
// I0, I1 : input frames (CV_8UC3 or CV_8UC1), same size
// vs     : output symmetric flow field (CV_32FC2)
// returns: true on success

bool computeSymmetricFlowTVL1(const cv::Mat& I0, const cv::Mat& I1, cv::Mat& vs)
{
    if (I0.empty() || I1.empty()) {
        std::cerr << "Error: one or both input frames are empty.\n";
        return false;
    }
    if (I0.size() != I1.size()) {
        std::cerr << "Error: input frames must have same size.\n";
        return false;
    }

    // Convert to grayscale for optical flow
    cv::Mat g0, g1;
    if (I0.channels() == 3)
        cv::cvtColor(I0, g0, cv::COLOR_BGR2GRAY);
    else
        g0 = I0.clone();

    if (I1.channels() == 3)
        cv::cvtColor(I1, g1, cv::COLOR_BGR2GRAY);
    else
        g1 = I1.clone();

    // Create TV-L1 optical flow instances
    cv::Ptr<cv::optflow::DualTVL1OpticalFlow> tvl1 =
        cv::optflow::createOptFlow_DualTVL1();

    cv::Mat flow_f, flow_b;

    // Forward flow: I0 -> I1
    tvl1->calc(g0, g1, flow_f);
    // Backward flow: I1 -> I0
    tvl1->calc(g1, g0, flow_b);

    // Apply spatial regularization to both flows using original frames as guides
    jointBilateralRegularization(I0, flow_f, 5, 20.0, 20.0);
    jointBilateralRegularization(I1, flow_b, 5, 20.0, 20.0);

    // Build symmetric flow: midpoint motion = 0.5 * (forward - backward)
    vs.create(flow_f.size(), flow_f.type());
    for (int y = 0; y < vs.rows; ++y) {
        const cv::Point2f* pf = flow_f.ptr<cv::Point2f>(y);
        const cv::Point2f* pb = flow_b.ptr<cv::Point2f>(y);
        cv::Point2f* ps = vs.ptr<cv::Point2f>(y);
        for (int x = 0; x < vs.cols; ++x) {
            ps[x] = 0.5f * (pf[x] - pb[x]);
        }
    }
    return true;
}

// Compute symmetric flow v_s between I0 and I1 using Farnebäck optical flow.
// I0, I1 : input frames (CV_8UC3 or CV_8UC1), same size
// vs     : output symmetric flow field (CV_32FC2)
// returns: true on success
bool computeSymmetricFlowFarneback(const Mat& I0, const Mat& I1, Mat& vs)
{
    if (I0.empty() || I1.empty()) {
        std::cerr << "Error: one or both input frames are empty.\n";
        return false;
    }

    if (I0.size() != I1.size()) {
        std::cerr << "Error: input frames must have the same size.\n";
        return false;
    }

    // Convert to grayscale for optical flow
    Mat g0, g1;
    if (I0.channels() == 3)
        cvtColor(I0, g0, COLOR_BGR2GRAY);
    else
        g0 = I0.clone();

    if (I1.channels() == 3)
        cvtColor(I1, g1, COLOR_BGR2GRAY);
    else
        g1 = I1.clone();

    Mat flow_f, flow_b;

    // Forward flow: I0 -> I1
    calcOpticalFlowFarneback(
        g0, g1, flow_f,
        0.5,   // pyr_scale
        3,     // levels
        15,    // winsize
        3,     // iterations
        5,     // poly_n
        1.2,   // poly_sigma
        0      // flags
    );

    // Backward flow: I1 -> I0
    calcOpticalFlowFarneback(
        g1, g0, flow_b,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    );

    // Build symmetric flow:
    // v_s(x) = 0.5 * (v_f(x) - v_b(x))
    vs.create(flow_f.size(), flow_f.type());

    for (int y = 0; y < vs.rows; ++y) {
        const Point2f* pf = flow_f.ptr<Point2f>(y);
        const Point2f* pb = flow_b.ptr<Point2f>(y);
        Point2f*       ps = vs.ptr<Point2f>(y);

        for (int x = 0; x < vs.cols; ++x) {
            Point2f vf = pf[x];
            Point2f vb = pb[x];
            ps[x] = 0.5f * (vf - vb);
        }
    }

    return true;
}
// Compute Linear Optical Flow between I0 and I1 using Farnebäck optical flow(Just forward flow).
// I0, I1 : input frames (CV_8UC3 or CV_8UC1), same size
// vs     : output forward flow field (CV_32FC2)
// returns: true on success
bool computeLinearFlowFarneback(const Mat& I0, const Mat& I1, Mat& vs)
{
    if (I0.empty() || I1.empty()) {
        std::cerr << "Error: one or both input frames are empty.\n";
        return false;
    }

    if (I0.size() != I1.size()) {
        std::cerr << "Error: input frames must have the same size.\n";
        return false;
    }

    // Convert to grayscale for optical flow
    Mat g0, g1;
    if (I0.channels() == 3)
        cvtColor(I0, g0, COLOR_BGR2GRAY);
    else
        g0 = I0.clone();

    if (I1.channels() == 3)
        cvtColor(I1, g1, COLOR_BGR2GRAY);
    else
        g1 = I1.clone();

    // Forward flow: I0 -> I1
    calcOpticalFlowFarneback(
        g0, g1, vs,
        0.5,   // pyr_scale
        3,     // levels
        15,    // winsize
        3,     // iterations
        5,     // poly_n
        1.2,   // poly_sigma
        0      // flags
    );

    return true;
}

// Bilinear sample with bounds check (expects CV_8UC3)
// this is responsible for bilinear sampling of a frame at floating point coordinates
// frame : input frame
// x, y  : floating point coordinates
// outPixel : output pixel
// returns  : true if sample is valid, false if out of bounds
static bool sampleFrameBilinear(const Mat& frame, float x, float y, Vec3b& outPixel)
{
    if (x < 0.0f || y < 0.0f || x >= frame.cols - 1.0f || y >= frame.rows - 1.0f)
        return false;

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    float dx = x - x0;
    float dy = y - y0;

    const Vec3b& p00 = frame.at<Vec3b>(y0,     x0);
    const Vec3b& p01 = frame.at<Vec3b>(y0,     x0 + 1);
    const Vec3b& p10 = frame.at<Vec3b>(y0 + 1, x0);
    const Vec3b& p11 = frame.at<Vec3b>(y0 + 1, x0 + 1);

    Vec3f r0 = p00 * (1.0f - dx) + p01 * dx;
    Vec3f r1 = p10 * (1.0f - dx) + p11 * dx;
    Vec3f r  = r0 * (1.0f - dy) + r1 * dy;

    outPixel = Vec3b(
        saturate_cast<uchar>(r[0]),
        saturate_cast<uchar>(r[1]),
        saturate_cast<uchar>(r[2])
    );

    return true;
}
