#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("Haze5.jpg", IMREAD_UNCHANGED); // Read in the image file

	// Contrast Stretch

	// First build a LUT
	unsigned char lut[256];
	double minVal, maxVal;
	minMaxLoc(img, &minVal, &maxVal); //find minimum and maximum intensities
	for (int i = 0; i < 256; i++)
	{
		lut[i] = 255*((i/(maxVal - minVal)) - (minVal / (maxVal - minVal)));
	}

	// Convert the image type to YCrCb
	Mat YCrCb;
	cvtColor(img, YCrCb, CV_BGR2YCrCb);
	vector<Mat> channels;
	split(YCrCb, channels);

	// Establish the number of bins
	int histSize = 256;

	// Set the range
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hist;

	// Compute the histogram
	calcHist(&channels[0], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

	// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	// Display and Save
	namedWindow("Original Image Histogram", CV_WINDOW_AUTOSIZE);
	imshow("Original Image Histogram", histImage);
	imwrite("Original_Image_Histogram.jpg", histImage);

	// Now stretch the contrast
	for (int i = 0; i < channels[0].rows; i++)
	{
		// Create a pointer to the pixel array
		uchar *p = channels[0].ptr<uchar>(i);
		for (int j = 0; j < channels[0].cols; j++)
		{
			p[j] = lut[p[j]];
		}
	}

	// Compute the histogram
	calcHist(&channels[0], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	Mat histEnhanced(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

	// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histEnhanced.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histEnhanced, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	// Display and Save
	namedWindow("Enhance Image Histogram", CV_WINDOW_AUTOSIZE);
	imshow("Enhance Image Histogram", histEnhanced);
	imwrite("Enhanced_Image_Histogram.jpg", histEnhanced);

	merge(channels, YCrCb);
	cvtColor(YCrCb, img, CV_YCrCb2BGR);

	namedWindow("Image Window", WINDOW_AUTOSIZE); // Create a window of the same size as the image for display
	imshow("Image Window", img); // Show our image inside the window
	imwrite("image.jpg", img);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}