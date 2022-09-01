#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void stereoCameraCalib(string checkerBoard_filePath_L, string checkerBoard_filePath_R, int boardSize_X, int boardSize_Y, float grid, bool mode = false)
{
	// read in images
	string archivePath_L = checkerBoard_filePath_L; // the file path of the left camera image archive
	string archivePath_R = checkerBoard_filePath_R; // the file path of the right camera image archive
	float gridLength = grid; // the length of a square on the checkerboard
	vector<String> imagePath_L; // the file paths of each left camera image
	vector<String> imagePath_R; // the file paths of each right camera image
	glob(archivePath_L, imagePath_L, true);
	glob(archivePath_R, imagePath_R, true);

	// star calibration
		vector<vector<Point3f>> objectPoints; // stores real world object coordinate points
		vector<vector<Point2f>> imagePoints_L; // stores image coordinate points of the left camera
		vector<vector<Point2f >> imagePoints_R; // stores image coordinates points of the right camera
		vector<Point3f> objPoints; // random real world coordinate points
		// allocate random coordinate points
		for (int i = 0; i < boardSize_X; i++)
			for (int j = 0; j < boardSize_Y; j++)
				objPoints.push_back(Point3f(j * gridLength, i * gridLength, 0));

		// find corner coordinates on each image
		Mat rgbImage_L, rgbImage_R; // RGB image matrixs
		Mat grayImage_L, grayImage_R; // gray level image matrixs
		vector<Point2f> cornerPoints_L, cornerPoints_R; // corner coordinate points on images
		for (int i = 0; i < imagePath_L.size(); i++)
		{
			rgbImage_L = imread(imagePath_L[i]);
			rgbImage_R = imread(imagePath_R[i]);
			// convert RGB image to gray level image
			cvtColor(rgbImage_L, grayImage_L, COLOR_BGR2GRAY);
			cvtColor(rgbImage_R, grayImage_R, COLOR_BGR2GRAY);
			bool successL = findChessboardCorners(grayImage_L, Size(boardSize_Y, boardSize_X), cornerPoints_L, \
				CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_ADAPTIVE_THRESH);
			bool successR = findChessboardCorners(grayImage_R, Size(boardSize_Y, boardSize_X), cornerPoints_R, \
				CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_ADAPTIVE_THRESH);
			if (successL && successR)
			{
				// refine corner coordinate points
				TermCriteria criteria(1|2, 200, 0.0001);
				cornerSubPix(grayImage_L, cornerPoints_L, Size(11, 11), Size(-1, 1), criteria);
				cornerSubPix(grayImage_R, cornerPoints_R, Size(11, 11), Size(-1, 1), criteria);
				drawChessboardCorners(rgbImage_L, Size(boardSize_Y, boardSize_X), cornerPoints_L, successL);
				drawChessboardCorners(rgbImage_R, Size(boardSize_Y, boardSize_X), cornerPoints_R, successR);

				objectPoints.push_back(objPoints);
				imagePoints_L.push_back(cornerPoints_L);
				imagePoints_R.push_back(cornerPoints_R);
				//cout << "left" + to_string(i) << cornerPoints_L << endl;
				//cout << "right" + to_string(i) << cornerPoints_R << endl;
				//cout << endl;
			}
			namedWindow("Left Display ", WINDOW_NORMAL);
			namedWindow("Right Display ", WINDOW_NORMAL);
			imshow("Left Display ", rgbImage_L);
			imshow("Right Display ", rgbImage_R);
			if(mode == true)  waitKey(0);
			else waitKey(1000);
		}
		cv::destroyAllWindows();
		cout << "Finish finding checkerboard corners" << endl;

		// calibrate left and right camera respectively
		Mat intrinsicMat_L, distortionMat_L, rvec_L, tvec_L;
		Mat intrinsicMat_R, distortionMat_R, rvec_R, tvec_R;
		Mat newIntrinsicMat_L, newIntrinsicMat_R;
		double RMS_L = calibrateCamera(objectPoints, imagePoints_L, Size(grayImage_L.cols, grayImage_L.rows), \
			intrinsicMat_L, distortionMat_L, rvec_L, tvec_L);
		cout << "Finish calibrate left camera" << endl;
		double RMS_R = calibrateCamera(objectPoints, imagePoints_R, Size(grayImage_R.cols, grayImage_R.rows), \
			intrinsicMat_R, distortionMat_R, rvec_R, tvec_R);
		cout << "Finish calibrate right camera" << endl;

		cout << "Left Camera RMS: " << RMS_L << endl;
		cout << "Right Camera RMS: " << RMS_R << endl;
		cout << "Finish individual calibrate process" << endl;

		/*
		* Stereo camera calibtration
		* compute rotation and translation matrix between the two camera
		*/     
		TermCriteria stereoCriteria(1 | 2, 200, 1e-6);
		Mat rotationMat_stereo, translationMat_stereo, essentialMat, fundamentalMat;
		double RMS_stereo = stereoCalibrate(objectPoints, imagePoints_L, imagePoints_R, intrinsicMat_L, distortionMat_L, \
			intrinsicMat_R, distortionMat_R, Size(grayImage_L.cols, grayImage_L.rows), rotationMat_stereo, translationMat_stereo, \
			essentialMat, fundamentalMat, CALIB_FIX_INTRINSIC | CALIB_SAME_FOCAL_LENGTH, stereoCriteria);
		cout << "Stereo RMS: " << RMS_stereo << endl;

		// Stereo Rectification
		Mat rect_L, rect_R, proj_rect_L, proj_rect_R, Q;
		stereoRectify(intrinsicMat_L, distortionMat_L, intrinsicMat_R, distortionMat_R, Size(grayImage_L.cols, grayImage_L.rows), \
			rotationMat_stereo, translationMat_stereo, rect_L, rect_R, proj_rect_L, proj_rect_R, Q, CALIB_ZERO_DISPARITY, 0);
		cout << "Finish stereo rectify process" << endl;

		cout << "Left Intrinsic Mat: " << endl;
		cout << intrinsicMat_L << endl;
		cout << "Right Intrinsic Mat: " << endl;
		cout << intrinsicMat_L << endl;
		cout << "Left Distortion Mat: " << endl;
		cout << distortionMat_L << endl;
		cout << "Right Distortion Mat: " << endl;
		cout << distortionMat_R << endl;
		cout << "Essential Mat: " << endl;
		cout << essentialMat << endl;
		cout << "Fundamental  Mat: " << endl;
		cout << fundamentalMat << endl;
		cout << "Q Mat: " << endl;
		cout << Q << endl;

		// get rectifiy maps
		Mat mapL_1, mapR_1, mapL_2, mapR_2;
		cv::initUndistortRectifyMap(intrinsicMat_L, distortionMat_L, rect_L, proj_rect_L, Size(grayImage_L.cols, grayImage_L.rows), CV_16SC2, mapL_1, mapR_1);
		cv::initUndistortRectifyMap(intrinsicMat_R, distortionMat_R, rect_R, proj_rect_R, Size(grayImage_R.cols, grayImage_R.rows), CV_16SC2, mapL_2, mapR_2);

		// save stereo calibration results
		string fileName = "output.yaml";
		FileStorage fs(fileName, FileStorage::WRITE);
		if (!fs.isOpened())
			cout << "Fail to open file" << endl;
		else
		{
			fs << "Left Cam Intrinsic Matrix" << intrinsicMat_L;
			fs << "Right Cam Intrinsic Matrix" << intrinsicMat_R;
			fs << "Left Cam Distortion Matrix" << distortionMat_L;
			fs << "Right Cam Distortion Matrix" << distortionMat_R;
			fs << "Essential Matrix" << essentialMat;
			fs << "Fundamental Matrix" << fundamentalMat;
			fs << "Disparity-to-Depth Mapping Matrix" << Q;
			fs << "mapL_1" << mapL_1;
			fs << "mapR_1" << mapR_1;
			fs << "mapL_2" << mapL_2;
			fs << "mapR_2" << mapR_2;
		}
		fs.release();	

		cout << "Finish stereo camera calibration process" << endl;

		// verify 
		Mat distorted_img_L, distorted_img_R;
		Mat undistorted_img_L, undistorted_img_R;
		FileStorage rfs(fileName, FileStorage::READ);
		if (!rfs.isOpened())
			cout << "Fail to open file" << endl;
		else
		{
			rfs["mapL_1"] >> mapL_1;
			rfs["mapR_1"] >> mapR_1;
			rfs["mapL_2"] >> mapL_2;
			rfs["mapR_2"] >> mapR_2;
			for (int i = 0; i < imagePath_L.size(); i++)
			{
				// undistort chessboard images for testing;
				distorted_img_L = imread(imagePath_L[i]);
				distorted_img_R = imread(imagePath_R[i]);
				remap(distorted_img_L, undistorted_img_L, mapL_1, mapR_1, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
				remap(distorted_img_R, undistorted_img_R, mapL_2, mapR_2, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

				namedWindow("Left Undistorted Image", WINDOW_NORMAL);
				namedWindow("Right Undistorted Image", WINDOW_NORMAL);
				imshow("Left Undistorted Image", undistorted_img_L);
				imshow("Right Undistorted Image", undistorted_img_R);
				imwrite("undistorted image\\left\\" + to_string(i) + ".jpg", undistorted_img_L);
				imwrite("undistorted image\\right\\" + to_string(i) + ".jpg", undistorted_img_R);
				waitKey(1000);

				// undistort left right single image 
				//distorted_img_L = imread("C:\\Users\\Aprilab\\Desktop\\Andy's File\\Master Degree\\ARHUD\\photos\\distorted\\20210512122635485.jpg");
				//distorted_img_R = imread("C:\\Users\\Aprilab\\Desktop\\Andy's File\\Master Degree\\ARHUD\\photos\\distorted\\20210512122640964.jpg");
				//remap(distorted_img_L, undistorted_img_L, mapL_1, mapR_1, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
				//remap(distorted_img_R, undistorted_img_R, mapL_2, mapR_2, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
				//namedWindow("Left Undistorted Image", WINDOW_NORMAL);
				//namedWindow("Right Undistorted Image", WINDOW_NORMAL);
				//imshow("Left Undistorted Image", undistorted_img_L);
				//imshow("Right Undistorted Image", undistorted_img_R);
				//imwrite("C:\\Users\\Aprilab\\Desktop\\Andy's File\\Master Degree\\ARHUD\\photos\\undistorted\\validate_left.jpg", undistorted_img_L);
				//imwrite("C:\\Users\\Aprilab\\Desktop\\Andy's File\\Master Degree\\ARHUD\\photos\\undistorted\\validate_right.jpg", undistorted_img_R);
				//waitKey(0);

				// undistort video
				//string clipPath = "C:\\Users\\Aprilab\\Desktop\\Andy's File\\Master Degree\\ARHUD\\Videos\\raw video\\under gateway\\right.avi";
				//VideoCapture clip1(clipPath.c_str());
				//VideoWriter saveVideo("Undistort Right.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 15, Size(1280, 960));
				//Mat frame, undistortFrame;
				//while (clip1.read(frame) == true)
				//{
				//	remap(frame, undistortFrame, mapL_1, mapR_1, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
				//	saveVideo << undistortFrame;
				//}
			}
		}
}

// int main()
//{
////	 calibrate stereo camera
//	string filePath_L = "checkerBoard (for testing)\\left\\*.jpg"; // path of the folder containing left camera images
//	string filePath_R = "checkerBoard (for testing)\\right\\*.jpg"; // path of the folder containing right camera images
//	int size_X = 8;
//	int size_Y = 6;
//	float gridLength = 39;
//	bool mode = false;
//	stereoCameraCalib(filePath_L, filePath_R, size_X, size_Y, gridLength, mode);
//}