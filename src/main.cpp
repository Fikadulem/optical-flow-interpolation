#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include "computeSymmetricFlow.h"
#include "spatialRegularization.h"
#include "warpUtils.h"
#include "occlusionHandling.h"
#include <iostream>
#include <fstream>
#include <filesystem>
using namespace cv;
using namespace std;

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


// Main function
int main(int argc, char** argv) {

    // Dataset folder path
    // std::string evalFolder = "/Users/fikadu.balcha/Downloads/data/eval_data/";
    // std::string gtFolder = "/Users/fikadu.balcha/Downloads/data/ground_truth/";
    // std::string interpFolder = "/Users/fikadu.balcha/Downloads/data/interpolated/";

    filesystem::path exePath = filesystem::canonical(filesystem::path(argv[0]));
    filesystem::path buildDir = exePath.parent_path();
    filesystem::path repoRoot = buildDir.parent_path();
    filesystem::path dataRoot = repoRoot / "inputframes";
    if (!filesystem::exists(dataRoot)) {
        dataRoot = repoRoot / "data";
    }

    string evalFolder = (dataRoot / "eval_data").string() + "/";
    string gtFolder = (dataRoot / "ground_truth").string() + "/";
    string interpFolder = (dataRoot / "interpolated").string() + "/";

    // print header once
    std::cout << std::left << std::setw(20) << "Dataset"
              << " | MAIE: " << std::setw(4) << ""
              << " | PSNR: " << std::setw(2)  << ""
              << " | SSIM: " << "" << std::endl;
    std::cout << std::string(18, '-') << "+" << std::string(13, '-') << "+" << std::string(11, '-') << "+" << std::string(8, '-') << std::endl;

    // Open results file
    std::ofstream results("/Users/fikadu.balcha/Downloads/results.txt", std::ios::out);
    if (!results) {
        std::cerr << "Warning: could not open results file" << std::endl;
    }

    for (auto& entry : std::filesystem::directory_iterator(gtFolder)) {
        if (!entry.is_directory()) continue;
        if (entry.path().filename() == ".DS_Store") continue;
        String path = evalFolder + entry.path().filename().string() + "/frame10.png";
        Mat frame10 = imread(evalFolder + entry.path().filename().string() + "/frame10.png");
        Mat frame11 = imread(evalFolder + entry.path().filename().string() + "/frame11.png");
        Mat frame10i11 = imread(gtFolder + entry.path().filename().string() + "/frame10i11.png");
        if (frame10.empty() || frame11.empty() || frame10i11.empty()) {
            std::cerr << "Error reading input image from folder " << entry.path().filename().string() << ".Skipping.\n";
            continue;
        }
        // Interpolate without Spatial regularization and Occlution handling
        Mat vsRaw;
        if (!computeSymmetricFlowFarneback(frame10, frame11, vsRaw)) {
            std::cerr << "Failed to compute raw symmetric flow.\n";
            return -1;
        }
        Mat interpRaw = interpolateSymmetric(frame10, frame11, vsRaw);
        if (interpRaw.empty()) {
            std::cerr << "Interpolation produced empty result.\n";
            continue;
        }
        // Create output directtory per dataset under the interpolated folder
        std::string dataset = entry.path().filename().string();
        std::string outDir = interpFolder + dataset + "/";
        std::error_code ec;
        std::filesystem::create_directories(outDir, ec);
        if (ec) {
            std::cerr << "Failed to create output directory " << outDir << ": " << ec.message() <<std::endl;
        }
        // Save interpolated image
        imwrite(outDir + "mid_raw.png", interpRaw);

        // Compute metrics for (interpolated without spatial regularization and occlusion handling)
        double maieBefore = computeMAIE(interpRaw, frame10i11);
        double psnrBefore = computePSNR(interpRaw, frame10i11);
        double ssimBefore = computeSSIM(interpRaw, frame10i11);

        // Print metrics
        auto printRow = [&] (std::ostream& os, const std::string& label, double maie, double psnr, double ssim) {
            os << std::left << std::setw(20) << label
               << std::right << std::setw(8) << std::fixed << std::setprecision(4) << maie << "  "
               << std::right << std::setw(8) << std::fixed << std::setprecision(4) << psnr << "  "
               << std::right << std::setw(8) << std::fixed << std::setprecision(4) << ssim 
               << std::endl;
        };

        printRow(std::cout, dataset, maieBefore, psnrBefore, ssimBefore);
        if (results) printRow(results, dataset, maieBefore, psnrBefore, ssimBefore);

        // // After SpatialRegularization and occlusion handling applied
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


        // Compute metrics after (Spatial regularized and + Occlusion handling/aware)
        double maieAfter = computeMAIE(interpReg, frame10i11);
        double psnrAfter = computePSNR(interpReg, frame10i11);
        double ssimAfter = computeSSIM(interpReg, frame10i11);


        // Save interpolated image with Spatial regularized and + Occlusion handling/aware
        imwrite(outDir + "mid_reg.png", interpReg);

        // Print rows for after Spatial regularized and + Occlusion handling/aware
        printRow(std::cout, dataset + " (after)", maieAfter, psnrAfter, ssimAfter);

    }
    if (results) results.close();
}