#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("C:\\George Seif\\rgb_squares.png", IMREAD_UNCHANGED); // Read in the image file
	int ch = img.channels();

	// For Colour Images
	for (int i = 0; i < img.rows; i++)
	{
		// Create a pointer to the pixel array
		uchar *p = img.ptr<uchar>(i);
		for (int j = 0; j < img.cols; j++)
		{
			p[j*ch + 0] = 0; // B
			p[j*ch + 1] = 0; // G
			p[j*ch + 2] = p[j*ch + 2]; // R
		}
	}

	/*for (int i = 0; i < 720; i++)
	{
		for (int j = 0; j < 720; j++)
		{
			img.at<Vec3b>(i, j)[0] = 0; // B
			img.at<Vec3b>(i, j)[1] = 0; // G
			img.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i, j).val[2]; // R
		}
	}*/
	
	// For Grayscale Images

	/*
	for (int i = 0; i < img.rows; i++)
	{
		// Create a pointer to the pixel array
		uchar *p = img.ptr<uchar>(i);
		for (int j = 0; j < img.cols; j++)
		{
			if (p[j] < 100){ p[j] = 0; }
			else if (p[j] > 155){ p[j] = 255; }
			else { p[j] = p[j]; }
		}
	}*/

	/*float A = 1.5;
	int B = 40;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<uchar>(i, j) = saturate_cast<uchar>((int)A*img.at<uchar>(i, j) + B);
		}
	}*/

	namedWindow("Image Window", WINDOW_AUTOSIZE); // Create a window of the same size as the image for display
	imshow("Image Window", img); // Show our image inside the window
	imwrite("image.jpg", img);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}