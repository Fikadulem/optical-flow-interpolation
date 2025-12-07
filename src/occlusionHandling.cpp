#include "occlusionHandling.h"
using namespace cv;

Mat computeOcclusionMask(const Mat &flowFwd,
                         const Mat &flowBwd,
                         float threshold) {
    CV_Assert(flowFwd.type() == CV_32FC2 && flowBwd.type() == CV_32FC2);
    CV_Assert(flowFwd.size() == flowBwd.size());

    Mat negFlowBwd = -flowBwd;

    Mat diffFlow;
    absdiff(flowFwd, negFlowBwd, diffFlow);

    // Compute magnitude of difference
    std::vector<cv::Mat> channels;
    split(diffFlow, channels);

    Mat diffMag;
    magnitude(channels[0], channels[1], diffMag);

    // Pixels with difference below threshold 
    Mat mask = (diffMag < threshold);
    mask.convertTo(mask, CV_32FC1); // convert to flaot or blending operations
    return mask;
}

// Compute occlusion mask by estimating forward/backward Farneback flows
Mat computeOcclusionMaskFarneback(const Mat& I0, const Mat& I1, float threshold) {
    CV_Assert(!I0.empty() && !I1.empty());
    CV_Assert(I0.size() == I1.size());

    //Grayscale for flow
    Mat g0, g1;
    if (I0.channels() == 3) cvtColor(I0, g0, COLOR_BGR2GRAY); else g0 = I0.clone();
    if (I1.channels() == 3) cvtColor(I1, g1, COLOR_BGR2GRAY); else g1 = I1.clone();

    Mat flowFwd, flowBwd;
    calcOpticalFlowFarneback(g0, g1, flowFwd, 0.5, 3, 15, 3, 5, 1.2, 0);
    calcOpticalFlowFarneback(g1, g0, flowBwd, 0.5, 3, 15, 3, 5, 1.2, 0);

    return computeOcclusionMask(flowFwd, flowBwd, threshold);
}