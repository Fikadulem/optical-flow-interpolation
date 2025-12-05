#include "computeSymmetricFlow.h"
#include "spatialRegularization.h"
#include <opencv2/optflow.hpp>
#include <iostream>

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