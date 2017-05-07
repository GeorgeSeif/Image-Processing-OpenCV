#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

int main(int argc, const char** argv)
{
	//Load the images
	Mat style_img = imread("paint.jpg");
	Mat tar_img = imread("clock_tower.jpg");

	// Convert to Lab space and CV_32F1
	Mat style_lab, tar_lab;
	cvtColor(style_img, style_lab, COLOR_BGR2Lab);
	cvtColor(tar_img, tar_lab, COLOR_BGR2Lab);
	style_lab.convertTo(style_lab, CV_32FC1);
	tar_lab.convertTo(tar_lab, CV_32FC1);

	//Find mean and std of each channel for each image
	Mat mean_style, mean_tar, std_dev_style, std_dev_tar;
	meanStdDev(style_lab, mean_style, std_dev_style);
	meanStdDev(tar_lab, mean_tar, std_dev_tar);

	// Split into individual channels
	std::vector<Mat> style_chs, tar_chs;
	split(style_lab, style_chs);
	split(tar_lab, tar_chs);

	// For each channel calculate the color distribution
	for (int i = 0; i < 3; i++) 
	{
		// Normalize the colour channels by subtracting the mean a dividing by the standard deviation
		tar_chs[i] -= mean_tar.at<double>(i);
		tar_chs[i] /= std_dev_tar.at<double>(i);

		// Apply the colour style of the style image by reversing the operation, 
		// multiplying by the standard deviation and adding the mean of the style image
		tar_chs[i] *= std_dev_style.at<double>(i);
		tar_chs[i] += mean_style.at<double>(i);
	}

	// Merge the channels, convert to CV_8UC1 each channel and convert to BGR
	Mat out;
	merge(tar_chs, out);
	out.convertTo(out, CV_8UC1);
	cvtColor(out, out, COLOR_Lab2BGR);

	// Show images
	namedWindow("Style image", 0);
	imshow("Style image", style_img);
	namedWindow("Target image", 0);
	imshow("Target image", tar_img);
	namedWindow("Output image", 0);
	imshow("Output image", out);

	waitKey(0);

	return 0;
}