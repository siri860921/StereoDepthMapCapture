#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace cv;
using namespace std;

void strereoTakePhoto(string writePath, VideoCapture cam1 = VideoCapture(0, CAP_DSHOW), VideoCapture cam2 = VideoCapture(1, CAP_DSHOW), bool controlPanel = true)
{
	// check if all the cameras are opened
	cout << "Camera 1: " << cam1.isOpened() << endl;
	cout << "Camera 2: " << cam2.isOpened() << endl;
	if (!cam1.isOpened() || !cam2.isOpened())
	{
		printf("Error in opening camera.");
	}
	else
	{
		// set camera parameters
		cam1.set(CAP_PROP_FRAME_WIDTH, 1920);
		cam1.set(CAP_PROP_FRAME_HEIGHT, 1080);
		cam1.set(CAP_PROP_FPS, 15);
		//cam1.set(CAP_PROP_EXPOSURE, -5);

		cam2.set(CAP_PROP_FRAME_WIDTH, 1920);
		cam2.set(CAP_PROP_FRAME_HEIGHT, 1080);
		cam2.set(CAP_PROP_FPS, 15);
		//cam2.set(CAP_PROP_EXPOSURE, -5);

		// show camera parameters
		cout << "Cam1" << endl;
		cout << "Image Width: " << cam1.get(CAP_PROP_FRAME_WIDTH) << endl;
		cout << "Image Height: " << cam1.get(CAP_PROP_FRAME_HEIGHT) << endl;
		cout << "Image FPS: " << cam1.get(CAP_PROP_FPS) << endl;
		cout << "Image Brightness: " << cam1.get(CAP_PROP_BRIGHTNESS) << endl;
		cout << "Image Constrast: " << cam1.get(CAP_PROP_CONTRAST) << endl;
		cout << "Image Saturation: " << cam1.get(CAP_PROP_SATURATION) << endl;
		cout << "Image Hue: " << cam1.get(CAP_PROP_HUE) << endl;
		cout << "Image Gamma: " << cam1.get(CAP_PROP_GAMMA) << endl;
		cout << "Image Gain: " << cam1.get(CAP_PROP_GAIN) << endl;
		//cout << "Image White Balance: " << cam1.get(CAP_PROP_AUTO_WB) << endl;
		//cout << "Image Exposure: " << cam1.get(CAP_PROP_EXPOSURE);
		printf("\n");
		cout << "Cam2" << endl;
		cout << "Image Width: " << cam2.get(CAP_PROP_FRAME_WIDTH) << endl;
		cout << "Image Height: " << cam2.get(CAP_PROP_FRAME_HEIGHT) << endl;
		cout << "Image FPS: " << cam2.get(CAP_PROP_FPS) << endl;
		cout << "Image Brightness: " << cam2.get(CAP_PROP_BRIGHTNESS) << endl;
		cout << "Image Constrast: " << cam2.get(CAP_PROP_CONTRAST) << endl;
		cout << "Image Saturation: " << cam2.get(CAP_PROP_SATURATION) << endl;
		cout << "Image Hue: " << cam2.get(CAP_PROP_HUE) << endl;
		cout << "Image Gamma: " << cam2.get(CAP_PROP_GAMMA) << endl;
		cout << "Image Gain: " << cam2.get(CAP_PROP_GAIN) << endl;
		//cout << "Image White Balance: " << cam2.get(CAP_PROP_AUTO_WB) << endl;
		//cout << "Image Exposure: " << cam2.get(CAP_PROP_EXPOSURE);

		// manual camera parameter control panel
		cam1.set(CAP_PROP_SETTINGS, 1);
		cam2.set(CAP_PROP_SETTINGS, 1);

		// detect keydown to take photos
		int imageCounter = 0;
		while (1)
		{
			Mat frame1, frame2;
			cam1.read(frame1);
			cam2.read(frame2);
			resize(frame1, frame1, Size(1280, 960));
			resize(frame2, frame2, Size(1280, 960));
			namedWindow("Camera Frame 1", WINDOW_NORMAL | WINDOW_KEEPRATIO);
			namedWindow("Camera Frame 2", WINDOW_NORMAL | WINDOW_KEEPRATIO);
			imshow("Camera Frame 1", frame1);
			imshow("Camera Frame 2", frame2);
			char photoTrigger = waitKey(1);

			// press "p" key to save image
			if (photoTrigger == 'p' || photoTrigger == 'P')
			{
				imwrite(writePath + "/left/left_" + to_string(imageCounter) + ".jpg", frame1);
				imwrite(writePath + "\\right\\right_" + to_string(imageCounter) + ".jpg", frame2);
				imageCounter++;
			}
		}
	}
}

//int main()
//{
//	const string writePath = "C:\\Users\\Aprilab\\Desktop\\Andy's File\\Master Degree\\ARHUD\\checkerBoard (for testing)";
//	VideoCapture cam1 = VideoCapture(0, CAP_DSHOW);
//	VideoCapture cam2 = VideoCapture(1, CAP_DSHOW);
//	bool controlPanel = true;
//	strereoTakePhoto(writePath, cam1, cam2, controlPanel);
//}