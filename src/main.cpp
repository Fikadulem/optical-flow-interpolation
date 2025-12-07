#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include "computeSymmetricFlow.h"
#include "spatialRegularization.h"
#include "warpUtils.h"
#include "occlusionHandling.h"
#include <iostream>
using namespace cv;

// Compute Mean Absolute Interpolation Error
double computeMAIE(const cv::Mat& pred, const cv::Mat& gt) {
    CV_Assert(pred.size() == gt.size());
    CV_Assert(pred.type() == gt.type());

    Mat diff;
    absdiff(pred, gt, diff);
    Scalar mae = mean(diff);

    double maie = 0.0;
    for (int i = 0; i<diff.channels(); ++i) {
        maie += mae[i];
    }
    return maie / diff.channels();

    }

// Compute Peak Signal-to-Noise Ratio
double computePSNR(const cv::Mat& pred, const cv::Mat& gt) {
    CV_Assert(pred.size() == gt.size());
    CV_Assert(pred.type() == gt.type());

    cv::Mat diff;
    cv::absdiff(pred, gt, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    cv::Scalar sse = cv::sum(diff);
    double mse = 0.0;
    for (int i = 0; i < diff.channels(); ++i) {
        mse += sse[i];
    }
    mse /= (double)(pred.total() * pred.channels());

    if (mse <= 1e-10) {
        return INFINITY; // No error
    } else {
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

// Compute Structural Similarity Index (SSIM)
double computeSSIM(const cv::Mat& pred, const cv::Mat& gt) {
    CV_Assert(pred.size() == gt.size());
    CV_Assert(pred.type() == gt.type());

    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat I1, I2;
    pred.convertTo(I1, CV_32F);
    gt.convertTo(I2, CV_32F);

    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    cv::Scalar mssim = cv::mean(ssim_map);

    double ssim = 0.0;
    for (int i = 0; i < ssim_map.channels(); ++i) {
        ssim += mssim[i];
    }
    return ssim / ssim_map.channels();
}


// main function
int main(int argc, char** argv) {

    if (argc < 4) {
        std::cerr << "Usage: " <<argv[0] << " <frame10> <frame11> <frame10i1>\n";
        return -1;
    }

    Mat frame10 = imread(argv[1]);
    Mat frame11 = imread(argv[2]);
    Mat frame10i11 = imread(argv[3]);
    if (frame10.empty() || frame11.empty() || frame10i11.empty()) {
        std::cerr << "Error reading input images.\n";
        return -1;
    }

    if (frame10.size() != frame11.size()) {
        std::cerr << "Input frames must have the same size.\n";
        return -1;
    }

    
    // Before spatial regulazirization 
    Mat vsRaw;
    if (!computeSymmetricFlowFarneback(frame10, frame11, vsRaw)) {
        std::cerr << "Failed to compute raw symmetric flow.\n";
        return -1;
    }
    Mat interpRaw = interpolateSymmetric(frame10, frame11, vsRaw);
    double maieRaw = computeMAIE(interpRaw, frame10i11);
    Mat interpolated = interpolateSymmetric(frame10, frame11, vsRaw);
    
    // Compute Mean Absolute Interpolation Error
    double maie = computeMAIE(interpolated, frame10i11);
    std::cout << "Mean Absolute Interpolation Error before Spatial Regularization and Occlusion Handling = " << maie << std::endl;

    // After SpatialRegularization
    Mat vsReg;
    Mat g0, g1;
    if (frame10.channels() ==3) cvtColor(frame10, g0, COLOR_BGR2GRAY); else g0 = frame10.clone();
    if (frame11.channels() ==3) cvtColor(frame11, g1, COLOR_BGR2GRAY); else g1 = frame11.clone();
    computeSymmetricFlowFarneback(frame10, frame11, vsReg);
    jointBilateralRegularization(g0, vsReg, 5, 20.0, 20.0);
    // Compute occlusion mask (forward-backward consistency)
    Mat occMask = computeOcclusionMaskFarneback(frame10, frame11, 1.0f);
    // Interpolate with occlusion awareness
    Mat interpReg = interpolateSymmetricWithOcclusion(frame10, frame11, vsReg, occMask);
    
    // Compute Mean Absolute Interpolation Error
    double maieReg = computeMAIE(interpReg, frame10i11);
    std::cout << "Mean Absolute Interpolation Error after Spatial Regularization and Occlusion Handling = " << maieReg << std::endl;
    interpolated = interpReg;

    double psnr = computePSNR(interpolated, frame10i11);
    double ssim = computeSSIM(interpolated, frame10i11);
    std::cout << "PSNR = " << psnr << " dB" << std::endl;
    std::cout << "SSIM = " << ssim << std::endl;
    // std::cout << "Mean Absolute Interpolation Error = " << maie << std::endl;
    
    // Display and save results  
    imshow("Frame 10", frame10);
    imwrite("frame10.jpg", frame10);
    imshow("Frame 11", frame11);
    imwrite("frame11.jpg", frame11);
    imshow("Interpolated Midpoint", interpolated);
    imwrite("interpolated_midpoint.jpg", interpolated);
    waitKey(0);

    return 0;
}