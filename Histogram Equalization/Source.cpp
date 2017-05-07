#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("lake.jpg", IMREAD_GRAYSCALE); // Read in the image file

	if (img.empty()){ "\nImage is empty!!\n"; }

	// Establish the number of bins
	int histSize = 256;

	// Set the range
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hist;

	// Compute the histogram
	calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

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

	namedWindow("Image Window", WINDOW_AUTOSIZE); // Create a window of the same size as the image for display
	imshow("Image Window", img); // Show our image inside the window
	imwrite("image.jpg", img);

	// Perform Global Histogram Equalization 
	Mat out;
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(2.0);
	clahe->apply(img, out);
	//equalizeHist(img, out);

	// Compute the histogram
	calcHist(&out, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	Mat outHistImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

	// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, outHistImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(outHistImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	// Display and Save
	namedWindow("Equalized Image Histogram", CV_WINDOW_AUTOSIZE);
	imshow("Equalized Image Histogram", outHistImage);
	imwrite("Equalized_Image_Histogram.jpg", outHistImage);

	namedWindow("Equalized Image Window", WINDOW_AUTOSIZE); // Create a window of the same size as the image for display
	imshow("Equalized Image Window", out); // Show our image inside the window
	imwrite("Equalized_Image.jpg", out);

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}