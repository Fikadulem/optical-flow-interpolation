#include <opencv2/opencv.hpp>
#include <iostream>
#include "spatialRegularization.h"
//#include <opencv2/optflow.hpp>

using namespace cv;

// Compute symmetric flow v_s between I0 and I1 using TV-L1.
// I0, I1 : input frames (CV_8UC3 or CV_8UC1), same size
// vs     : output symmetric flow field (CV_32FC2)
// returns: true on success
/*
bool computeSymmetricFlowTVL1(const Mat& I0, const Mat& I1, Mat& vs)
{
    if (I0.empty() || I1.empty()) {
        std::cerr << "Error: one or both input frames are empty.\n";
        return false;
    }

    if (I0.size() != I1.size()) {
        std::cerr << "Error: input frames must have the same size.\n";
        return false;
    }

    // Convert to grayscale for TV-L1
    Mat g0, g1;
    if (I0.channels() == 3)
        cvtColor(I0, g0, COLOR_BGR2GRAY);
    else
        g0 = I0.clone();

    if (I1.channels() == 3)
        cvtColor(I1, g1, COLOR_BGR2GRAY);
    else
        g1 = I1.clone();

    // Create TV-L1 optical flow object
    Ptr<optflow::DualTVL1OpticalFlow> tvl1 = optflow::createOptFlow_DualTVL1();

    Mat flow_f, flow_b;

    // Forward flow: I0 -> I1
    tvl1->calc(g0, g1, flow_f);

    // Backward flow: I1 -> I0
    tvl1->calc(g1, g0, flow_b);

    // Apply spatial regularization to both flows using original frames as guides
    jointBilateralRegularization(I0, flow_f, 5, 20.0, 20.0);
    jointBilateralRegularization(I1, flow_b, 5, 20.0, 20.0);

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
*/
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

    // Apply spatial regularization to both flows using original frames as guides
    jointBilateralRegularization(I0, flow_f, 5, 20.0, 20.0);
    jointBilateralRegularization(I1, flow_b, 5, 20.0, 20.0);

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

//Forward Farneback optical flow
bool computeForwardFlowFarneback(const Mat& I0, const Mat& I1, Mat& vf)
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
        g0, g1, vf,
        0.5,   // pyr_scale
        3,     // levels
        15,    // winsize
        3,     // iterations
        5,     // poly_n
        1.2,   // poly_sigma
        0      // flags
    );

    // Apply spatial regularization to flow using original frame as guide
    jointBilateralRegularization(I0, vf, 5, 20.0, 20.0);
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

// Symmetric interpolation (no occlusion yet)
// I0, I1  : input RGB frames (CV_8UC3)
// vs      : symmetric flow field at middle time (CV_32FC2)
// returns : interpolated midpoint frame
Mat interpolateSymmetric(
    const Mat& I0,
    const Mat& I1,
    const Mat& vs)
{
    CV_Assert(I0.size() == I1.size());
    CV_Assert(I0.type() == CV_8UC3 && I1.type() == CV_8UC3);
    CV_Assert(vs.size() == I0.size() && vs.type() == CV_32FC2);

    Mat I_mid(I0.size(), CV_8UC3);

    const int W = I0.cols;
    const int H = I0.rows;

    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            // symmetric flow at the midpoint (motion to t=0.5)
            Point2f v = vs.at<Point2f>(y, x);

            // Coordinates in I0 and I1 along trajectories:
            // midpoint -> frame 0 : (x - v)
            // midpoint -> frame 1 : (x + v)
            float x0 = x - v.x;
            float y0 = y - v.y;

            float x1 = x + v.x;
            float y1 = y + v.y;

            Vec3b c0, c1;
            bool valid0 = sampleFrameBilinear(I0, x0, y0, c0);
            bool valid1 = sampleFrameBilinear(I1, x1, y1, c1);

            Vec3b out;

            if (valid0 && valid1)
            {
                // Ideal symmetric case: both samples valid
                out[0] = static_cast<uchar>((c0[0] + c1[0]) * 0.5f);
                out[1] = static_cast<uchar>((c0[1] + c1[1]) * 0.5f);
                out[2] = static_cast<uchar>((c0[2] + c1[2]) * 0.5f);
            }
            else if (valid0 && !valid1)
            {
                out = c0; // only I0 contributed
            }
            else if (!valid0 && valid1)
            {
                out = c1; // only I1 contributed
            }
            else
            {
                // both invalid (outside image) → fallback to simple average at (x,y)
                const Vec3b& p0 = I0.at<Vec3b>(y, x);
                const Vec3b& p1 = I1.at<Vec3b>(y, x);
                out[0] = static_cast<uchar>((p0[0] + p1[0]) * 0.5f);
                out[1] = static_cast<uchar>((p0[1] + p1[1]) * 0.5f);
                out[2] = static_cast<uchar>((p0[2] + p1[2]) * 0.5f);
            }

            I_mid.at<Vec3b>(y, x) = out;
        }
    }

    return I_mid;
}

// Forward-only interpolation 
// I0, I1  : input RGB frames (CV_8UC3)
// vf      : forward flow field from I0 to I1 (CV_32FC2)
// returns : interpolated midpoint frame
Mat interpolateForwardOnly(
    const Mat& I0,
    const Mat& I1,
    const Mat& vf)
{
    CV_Assert(I0.size() == I1.size());
    CV_Assert(I0.type() == CV_8UC3 && I1.type() == CV_8UC3);
    CV_Assert(vf.size() == I0.size() && vf.type() == CV_32FC2);

    Mat I_mid(I0.size(), CV_8UC3);

    const int W = I0.cols;
    const int H = I0.rows;

    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            // forward flow at the midpoint (motion to t=1.0)
            Point2f v = vf.at<Point2f>(y, x);

            // Coordinates in I0 along trajectory:
            // midpoint -> frame 0 : (x - 0.5 * v)
            float x0 = x - 0.5f * v.x;
            float y0 = y - 0.5f * v.y;

            Vec3b c0;
            bool valid0 = sampleFrameBilinear(I0, x0, y0, c0);

            Vec3b out;

            if (valid0)
            {
                out = c0; // only I0 contributed
            }
            else
            {
                // invalid (outside image) → fallback to simple average at (x,y)
                const Vec3b& p0 = I0.at<Vec3b>(y, x);
                const Vec3b& p1 = I1.at<Vec3b>(y, x);
                out[0] = static_cast<uchar>((p0[0] + p1[0]) * 0.5f);
                out[1] = static_cast<uchar>((p0[1] + p1[1]) * 0.5f);
                out[2] = static_cast<uchar>((p0[2] + p1[2]) * 0.5f);
            }

            I_mid.at<Vec3b>(y, x) = out;
        }
    }

    return I_mid;
}