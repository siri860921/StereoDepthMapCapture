#include <stdio.h>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <time.h>
#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace std;
using namespace dnn;
using namespace cuda;

Mat worldCoordinate;

void Optimize(cv::Mat &depth, int minDisparity)
{

    const int width = depth.cols;
    const int height = depth.rows;
    float* data = (float*)depth.data;
    cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
    double* integral = (double*)integralMap.data;
    int* ptsIntegral = (int*)ptsMap.data;
    memset(integral, 0, sizeof(double) * width * height);
    memset(ptsIntegral, 0, sizeof(int) * width * height);
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            if (data[id2] > 255)
            {
                integral[id2] = data[id2];
                ptsIntegral[id2] = 1;
            }
        }
    }
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 1; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - 1];
            ptsIntegral[id2] += ptsIntegral[id2 - 1];
        }
    }
    for (int i = 1; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - width];
            ptsIntegral[id2] += ptsIntegral[id2 - width];
        }
    }
    int wnd;
    double dWnd =81;
    while (dWnd > 1)
    {
        wnd = int(dWnd);
        dWnd /= 2;
        for (int i = 0; i < height; ++i)
        {
            int id1 = i * width;
            for (int j = 0; j < width; ++j)
            {
                int id2 = id1 + j;
                int left = j - wnd - 1;
                int right = j + wnd;
                int top = i - wnd - 1;
                int bot = i + wnd;
                left = max(0, left);
                right = min(right, width - 1);
                top = max(0, top);
                bot = min(bot, height - 1);
                int dx = right - left;
                int dy = (bot - top) * width;
                int idLeftTop = top * width + left;
                int idRightTop = idLeftTop + dx;
                int idLeftBot = idLeftTop + dy;
                int idRightBot = idLeftBot + dx;
                int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
                double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
                if (ptsCnt <= 0)
                {
                    continue;
                }
                data[id2] = float(sumGray / ptsCnt);
            }
        }
        int s = wnd / 2 * 2 + 1;
        if (s > 201)
        {
            s = 201;
        }
        cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
    }
}

void onMouse(int event, int x, int y, int flag, void*)
{
    cv::Point origin;
    if (event == EVENT_LBUTTONDOWN)
    {
        origin = Point(x, y);
        std::cout << origin << "in world coordinate is: " << worldCoordinate.at<cv::Vec3f>(origin) << std::endl;
    }
}

void depthData(Mat leftMap1, Mat rightMap1, Mat leftMap2, Mat rightMap2, Mat Q, VideoCapture leftCap, VideoCapture rightCap)
{
    if (!leftCap.isOpened() || !rightCap.isOpened()) // check if the videos are available
        cout << "Missing video files!" << endl;
    else
    {
        // image settings
        Size imgSize = Size(512, 384);
        Mat frame1, frame2, uframe1, uframe2;
        cout << Q << endl;

        // set SGBM parameters
        int minDisparity = 0; //16;
        int numberOfDisparity = 128; //90;
        int blockSize = 5; // 3;
        int P1 = 8; // 8;
        int P2 = 128; // 128;
        int preFilterCap = 91; // 15;
        int disp12Maxdiff = -1; // -1;
        int uniquenessRatio = 1; // 1;
        int speckleRange = 100; // 100;
        int speckleWindowSize = 100; // 100;

        while (leftCap.read(frame1) && rightCap.read(frame2))
        {
            clock_t start_t, end_t;
            start_t = clock();
            // remap distorted image to undistorted image
            remap(frame1, uframe1, leftMap1, rightMap1, cv::INTER_LANCZOS4, BORDER_CONSTANT, Scalar(0, 0, 0));
            remap(frame2, uframe2, leftMap2, rightMap2, cv::INTER_LANCZOS4, BORDER_CONSTANT, Scalar(0, 0, 0));
            // resize the input image
            resize(uframe1, uframe1, imgSize, 0, 0); 
            resize(uframe2, uframe2, imgSize, 0, 0);
            // set ROI
            Mat left_ROI = uframe1(Rect(0, uframe1.cols * 0.15, uframe1.cols, uframe1.rows * 0.4));
            Mat right_ROI = uframe2(Rect(0, uframe2.cols * 0.15, uframe2.cols, uframe2.rows * 0.4));
            medianBlur(left_ROI, left_ROI, 3);
            medianBlur(right_ROI, right_ROI, 3);
            Mat disp, disp2, disparity, disparityImg;
            // create SGBM object
            Ptr<StereoMatcher> stereo = StereoSGBM::create(minDisparity, numberOfDisparity, blockSize, \
                P1 * 3 * blockSize * blockSize, P2 * 3 * blockSize * blockSize, disp12Maxdiff, preFilterCap, \
                uniquenessRatio, speckleWindowSize, speckleRange, StereoSGBM::MODE_HH);
            
            Ptr<ximgproc::DisparityWLSFilter> wlsFilter = ximgproc::createDisparityWLSFilter(stereo);
            Ptr<StereoMatcher> stereo2 = ximgproc::createRightMatcher(stereo);
            wlsFilter->setLambda(800);
            wlsFilter->setSigmaColor(1.5);
            wlsFilter->setDepthDiscontinuityRadius(3);
            wlsFilter->setLRCthresh(24);
            stereo->compute(left_ROI, right_ROI, disp);
            stereo2->compute(right_ROI, left_ROI, disp2);
            wlsFilter->filter(disp, left_ROI, disp, disp2, Rect(), right_ROI);

            disp.convertTo(disparity, CV_32F);
            Mat disparity_filtered;
            Optimize(disparity, minDisparity); // multi-level median filter optimization
            //bilateralFilter(disparity, disparity_filtered, 5, 20, 20);
            //disparityImg = (disparity / float(16) - float(minDisparity)) / float(numberOfDisparity);
            disparity.convertTo(disparityImg, CV_8U, 1.0 / 16.0);
            
            resize(disparity, disparity, Size(1280, 960));
            resize(disparityImg, disparityImg, Size(1280, 960));
            applyColorMap(disparityImg, disparityImg, COLORMAP_JET);
            reprojectImageTo3D(disparity, worldCoordinate, Q, true);
            worldCoordinate = worldCoordinate * 16 * 1.2;

            // show image
            //namedWindow("Left distorted", WINDOW_NORMAL);
            //imshow("Left distorted", frame1);
            //namedWindow("Right distorted", WINDOW_NORMAL);
            //imshow("Right distorted", frame2);
            namedWindow("Left Capture", WINDOW_KEEPRATIO);
            imshow("Left Capture", left_ROI);
            namedWindow("Right Capture", WINDOW_KEEPRATIO);
            imshow("Right Capture", right_ROI);
            namedWindow("Disparity Map", WINDOW_KEEPRATIO);
            imshow("Disparity Map", disparityImg);
            cv::setMouseCallback("Disparity Map", onMouse, 0);
            waitKey(1);
            end_t = clock();
            cout << "Spend time: " << (double)(end_t - start_t) / CLOCKS_PER_SEC << endl;
        }      
    }
}

int main()
{
    FileStorage fs("output.yaml", FileStorage::READ);  // read in left and right video files
    Mat mapL_1, mapR_1, mapL_2, mapR_2, Q;
    fs["mapL_1"] >> mapL_1;
    fs["mapR_1"] >> mapR_1;
    fs["mapL_2"] >> mapL_2;
    fs["mapR_2"] >> mapR_2;
    fs["Disparity-to-Depth Mapping Matrix"] >> Q;
	string left_video_path = "on gateway 2\\left1.avi";
    string right_video_path = "on gateway 2\\right2.avi";
	VideoCapture left_capture(left_video_path.c_str());
    VideoCapture right_capture(right_video_path.c_str());
    depthData(mapL_1, mapR_1, mapL_2, mapR_2, Q, left_capture, right_capture);
}


