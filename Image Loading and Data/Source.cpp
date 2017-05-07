#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("C:\\George Seif\\Woman.png", IMREAD_UNCHANGED); // Read in the image file

	// Print out some information about the image
	int rows = img.rows;
	int cols = img.cols;
	int channels = img.channels();

	cout << "Image Rows = " << rows << endl;
	cout << "Image Columns = " << cols << endl;
	cout << "Image Channels = " << channels << endl;

	namedWindow("Image Window", WINDOW_AUTOSIZE); // Create a window of the same size as the image for display
	imshow("Image Window", img); // Show our image inside the window
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}