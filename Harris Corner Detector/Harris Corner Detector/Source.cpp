/*
This is an implementation of the Harris Corner Detector from the paper :
Harris, Chris, and Mike Stephens. "A combined corner and edge detector."
Alvey vision conference.Vol. 15. 1988.

The algorithm steps are :
(1) Compute the horizontal and vertical derivatives of the image
(2) Compute 3 images corresponding to the 3 different terms in martix M
(3) Convolve each of these three images with a Gaussian to have spatial weighting
(4) Compute the cornerness value using the eigen values of matrix M
(5) Find the local maxima(Non - Maximum Suppression) above some threshold
to get the final corner points
*/

#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	// Input image
	Mat img = imread("corner_detection_test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img, CV_32F); // 32-bit float type

	float cornerThreshold = 0.02; // Cornerness threshold value

	// (1) First we compute the derivatives
	Mat grad_x, grad_y;
	Sobel(img.clone(), grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(img.clone(), grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);

	// (2) We need to find the matrix M which is defined as:
	// [X*X, X*Y; X*Y, Y*Y]
	// Where X and Y are the horizontal and vertical derivatives of the image, respectively
	Mat XX = Mat::zeros(Size(img.cols, img.rows), CV_32F),
		XY = Mat::zeros(Size(img.cols, img.rows), CV_32F),
		YY = Mat::zeros(Size(img.cols, img.rows), CV_32F);
	multiply(grad_x.clone(), grad_x.clone(), XX);
	multiply(grad_x.clone(), grad_y.clone(), XY);
	multiply(grad_y.clone(), grad_y.clone(), YY);

	// (3) Now we apply the Gaussian filter to have spatial weighting
	GaussianBlur(XX.clone(), XX, Size(7, 7), 1, 1);
	GaussianBlur(XY.clone(), XY, Size(7, 7), 1, 1);
	GaussianBlur(YY.clone(), YY, Size(7, 7), 1, 1);

	// (4) and(5) Now we build the matrix M, compute the cornerness R, and threshold it
	float alpha = 0.05;
	Mat M = Mat::zeros(Size(2, 2), CV_32F);
	vector<float> eigenvalues(2);
	Mat cornerness = Mat::zeros(Size(img.cols, img.rows), CV_32F);
	for (int i = 0; i < img.rows; i++)
	{
		// Create a pointer to the pixel array
		float *xx = XX.ptr<float>(i);
		float *xy = XY.ptr<float>(i);
		float *yy = YY.ptr<float>(i);
		for (int j = 0; j < img.cols; j++)
		{
			M.at<float>(0, 0) = xx[j];
			M.at<float>(0, 1) = xy[j];
			M.at<float>(1, 0) = xy[j];
			M.at<float>(1, 1) = yy[j];

			eigen(M, eigenvalues);

			cornerness.at<float>(i, j) = (float)determinant(M) - alpha*((float)trace(M)[0] * (float)trace(M)[0]);
			//cornerness.at<float>(i, j) = eigenvalues[0] * eigenvalues[1] - alpha*(eigenvalues[0] + eigenvalues[1])*(eigenvalues[0] + eigenvalues[1]);

		}
	}

	// Threshold
	threshold(cornerness, cornerness, 100000000, 255, THRESH_TOZERO);

	// Non-Maximum Suppression
	int rad = 3;
	int dim = 2 * rad + 1;

	for (int i = rad; i < img.rows - rad; i++) // Start looping slightly inward to handle the borders for sliding window
	{
		// Create a pointer to the pixel array
		float *p = cornerness.ptr<float>(i);

		for (int j = rad; j < img.cols - rad; j++)
		{
			Mat NMS = cornerness(Range(j - rad, j + rad), Range(i - rad, i + rad)); // Extract the sliding window
			double minVal, maxVal;
			minMaxLoc(NMS, &minVal, &maxVal);
			if (p[j] < maxVal)
			{
				p[j] = 0;
			}
		}
	}

	// Draw the corner points on the image
	for (int i = 0; i < img.rows; i++)
	{
		float *p = cornerness.ptr<float>(i);
		for (int j = 0; j < img.cols; j++)
		{
			if (p[j] > 0) { circle(img, Point(j, i), 1, Scalar(0, 0, 255)); }
		}
	}

	// Convert to uchar for display
	img.convertTo(img, CV_8U);
	imshow("Corners Image", img);

	waitKey(0);
	return 0;
}