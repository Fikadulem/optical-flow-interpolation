#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "computeSymmetricFlow.h"
#include "warpUtils.h"
#include <iostream>

// Compute Mean Absolute Interpolation Error
double computeMAIE(const cv::Mat& pred, const cv::Mat& gt) {
    CV_Assert(pred.size() == gt.size());
    CV_Assert(pred.type() == gt.type());

    cv::Mat diff;
    cv::absdiff(pred, gt, diff);
    cv::Scalar mae = cv::mean(diff);

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
int main() {
    const std::string path0 = "C:/Users/olade/optical-flow-interpolation/frame10.png";
    const std::string path1 = "C:/Users/olade/optical-flow-interpolation/frame11.png";
    const std::string pathGT = "C:/Users/olade/optical-flow-interpolation/frame10i11.png";

    cv::Mat frame10 = cv::imread(path0);
    cv::Mat frame11 = cv::imread(path1);
    cv::Mat frame10i11 = cv::imread(pathGT);
    if (frame10.empty() || frame11.empty() || frame10i11.empty()) {
        std::cerr << "Error reading input images." << std::endl;
        return -1;
    }

    if (frame10.size() != frame11.size()) {
        std::cerr << "Input frames must have the same size." << std::endl;
        return -1;
    }

    cv::Mat vs;
    if (!computeSymmetricFlowFarneback(frame10, frame11, vs)) {
        std::cerr << "Failed to compute symmetric flow." << std::endl;
        return -1;
    }

    cv::Mat interpolated = interpolateSymmetric(frame10, frame11, vs);

    double maie = computeMAIE(interpolated, frame10i11);
    double psnr = computePSNR(interpolated, frame10i11);
    double ssim = computeSSIM(interpolated, frame10i11);
    std::cout << "PSNR = " << psnr << " dB" << std::endl;
    std::cout << "SSIM = " << ssim << std::endl;
    std::cout << "Mean Absolute Interpolation Error = " << maie << std::endl;

    //cv::imwrite("frame10_out.jpg", frame10);
    //cv::imwrite("frame11_out.jpg", frame11);
    //cv::imwrite("interpolated_midpoint.jpg", interpolated);

    // Optional display; comment out if running headless
    cv::imshow("Frame 10", frame10);
    cv::imshow("Frame 11", frame11);
    cv::imshow("Interpolated Midpoint", interpolated);
    cv::waitKey(0);

    return 0;
}
