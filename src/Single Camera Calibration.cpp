#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include < time.h>

using namespace std;
using namespace cv;

/// <summary>
/// Camera calibration process.
/// </summary>
/// <param name="checkerBoard_filePath"> Checkerboard archive file path</param>
/// <param name="boardSize_X"> Number of intersections on the horizontal direction of the checkerboard</param>
/// <param name="boardSize_Y"> Number of intersections on the verticle direction of the checkerboard</param>
void Calibarate_Camera(string checkerBoard_filePath, int boardSize_X, int boardSize_Y)
{
	vector < vector<Point3f>> objectPoints; // to store the 3D space position points of each checkboard
	vector <vector<Point2f>> imgPoints; // to store the 2D image position points of each checkboard
	// defining the world coordinates for 3D points
	vector<Point3f> objPoints;
	for (int i = 0; i < boardSize_X; i++)
		for (int j = 0; j < boardSize_Y; j++)
			objPoints.push_back(Point3f(j, i, 0));

	// get the path of each individual image
	vector<String> images;
	string path = checkerBoard_filePath;
	Mat rgb_img; // rgb image array
	Mat gray_img; // gray image
	bool success;
	vector<Point2f> cornerPoints;
	glob(path, images, false);
	for (int i = 0; i < images.size(); i++)
	{
		// convert rgb image to gray image
		rgb_img = imread(images[i]);
		//resize(rgb_img, rgb_img, Size(1920, 1080));
		cvtColor(rgb_img, gray_img, COLOR_BGR2GRAY);
		// find the corners of a single check board image
		// if the desired number of corners are found, then set to true
		success = findChessboardCorners(gray_img, Size(boardSize_Y, boardSize_X), cornerPoints, CALIB_CB_FAST_CHECK);
		/*
		* if desired corners are detected,
		* refine the pixel coordinates and display
		* them on images of checkboard
		*/
		if (success)
		{
			cout << "Success" << endl; 
			TermCriteria criteria(2, 60, 0.0001);
			// refine pixel coordinates for given 2D points
			cornerSubPix(gray_img, cornerPoints, Size(11, 11), Size(-1, -1), criteria);
			// display the detected corners on checkboard
			drawChessboardCorners(rgb_img, Size(boardSize_Y, boardSize_X), cornerPoints, success);
			objectPoints.push_back(objPoints);
			imgPoints.push_back(cornerPoints);
		}
		namedWindow("Display",  WINDOW_NORMAL);
		imshow("Display", rgb_img);
		waitKey(1000);
	}
	cv::destroyAllWindows();

	// camera Calibration
	Mat intrinsicMat, distortionMat, rotationMat, rotationMat_Rodrigues, translationMat;
	double RMS = calibrateCamera(objectPoints,  imgPoints, Size(gray_img.cols, gray_img.rows), intrinsicMat, distortionMat, rotationMat_Rodrigues, translationMat);
	//Rodrigues(rotationMat_Rodrigues, rotationMat);
	/*cout << "Rotation Matrix(Rodrigues): " << rotationMat_Rodrigues << endl;*/
	//cout << "Rotation Matrix: " << rotationMat << endl;
	/*cout << "Translation Matrix: " << translationMat << endl;*/

	// refining intrinsic maxtrix
	Mat intrinsicMat_new;
	intrinsicMat_new = getOptimalNewCameraMatrix(intrinsicMat, distortionMat, Size(gray_img.cols, gray_img.rows), 0, Size(gray_img.cols, gray_img.rows));
	cout << "Finish Caluibration" << endl;
	cout << "Intrinsic Matrix: " << intrinsicMat << endl;
	cout << "Distortion Matrix: " << distortionMat << endl;
	cout << "RMS: " << RMS << endl;
	cout << "Rectified Intrinsic Matrix: " << intrinsicMat_new << endl;

	// get rectify maps
	Mat map1, map2;
	Mat distorted_img = imread("C:/Users/Aprilab/Desktop/Andy's File/Master Degree/ARHUD/checkerBoard/S__78028948.jpg", 1);
	cv::initUndistortRectifyMap(intrinsicMat, distortionMat, Mat::eye(3, 3, CV_32FC1), intrinsicMat_new, Size(distorted_img.cols, distorted_img.rows), CV_16SC2, map1, map2);

	// safe camera calibration results
	fstream resultFile;
	resultFile.open("C:\\Users\\Aprilab\\Desktop\\Andy's File\\Master Degree\\ARHUD\\Camera Calibration.txt", ios::out | ios::trunc);
	resultFile << "Intrinsic Matrix" << "\n";
	resultFile << intrinsicMat << "\n";
	resultFile << "Distorsion Matrix" << "\n";
	resultFile << distortionMat << "\n";
	resultFile << "Refined Intrinsic Matrix" << "\n";
	resultFile << intrinsicMat_new << "\n";
	resultFile << "RMS: " << RMS << "\n" << "\n";
	resultFile << "Map1" << "\n";
	resultFile << map1 << "\n";
	resultFile << "Map2" << "\n";
	resultFile << map2 << "\n";

	// undistort image for testing
	Mat  undistorted_img;
	//resize(distorted_img, distorted_img, Size(1920, 1080));
	//namedWindow("Distorted Image",  WINDOW_NORMAL);
	//imshow("Distorted Image", distorted_img);
	clock_t start_time = clock();
	remap(distorted_img, undistorted_img, map1, map2, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	clock_t end_time = clock();
	cout << (double)(end_time - start_time) / CLOCKS_PER_SEC << endl;
	imwrite("C:\\Users\\Aprilab\\Desktop\\1.jpg", undistorted_img);
	namedWindow("Undistorted Image", WINDOW_NORMAL);
	imshow("Undistorted Image", undistorted_img);
	waitKey(0);
	cv::destroyAllWindows();
}
//int main()
//{
//	// calibrate camera
//	string filePath = ""; // path of the folder containing images
//	int size_X = 8;
//	int size_Y = 6;
//	Calibarate_Camera(filePath, size_X, size_Y);
//	return 0;
//}

