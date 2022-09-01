#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;


void image2Clip_stereo(double fps, string imagePath_L, string imagePath_R, Size clipSize = Size(100, 100), int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v'))
{
	VideoWriter left_video("test_left.mp4", fourcc, fps, clipSize);
	VideoWriter right_video("test_right.mp4", fourcc, fps, clipSize);
	vector<String> left_fn;
    vector<string> right_fn;
	glob(imagePath_L, left_fn, false);
	glob(imagePath_R, right_fn, false);
	size_t count = left_fn.size();
    for (size_t i = 0; i < count; i++)
	{
		Mat left_image = imread(left_fn[i]);
		Mat right_image = imread(right_fn[i]);
;		resize(left_image, left_image, Size(360, 109));
        resize(right_image, right_image, Size(360, 109));
		left_video << left_image;
		right_video << right_image;
	}
}

//int main()
//{
//	double clipFPS = 20.0;
//	Size clipSize = Size(360, 109);
//	string imagePath_L = "";
//	string imagePath_R = "";
//	image2Clip_stereo(clipFPS, imagePath_L, imagePath_R, clipSize);
//}