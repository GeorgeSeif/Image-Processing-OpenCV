#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("tulips.png");
	Mat blur, imgLaplacian, imgCanny;
	vector<Mat> rgbColourEdges(3);
	
	// Smooth the image to reduce noise for more accurate edge detection
	GaussianBlur(img.clone(), blur, Size(3, 3), 0, 0, BORDER_DEFAULT);

	// Now for colour edge detection
	vector<Mat> channels(3);
	split(blur.clone(), channels);
	double T = 20;
	double maxval = 255;
	for (int i = 0; i < 3; i++)
	{
		// Compute the second-order derivate i.e Laplacian operator
		Laplacian(blur.clone(), rgbColourEdges[i], CV_32F);
		convertScaleAbs(rgbColourEdges[i], rgbColourEdges[i]);

		threshold(rgbColourEdges[i], rgbColourEdges[i], T, maxval, THRESH_BINARY);
		
	}

	// Logical operation to between colour channels
	Mat tempEdges;
	cv::bitwise_or(rgbColourEdges[0], rgbColourEdges[1], tempEdges);
	cv::bitwise_or(tempEdges, rgbColourEdges[2], imgLaplacian);
	
	// Convert pixel type for display
	imgLaplacian.convertTo(imgLaplacian, CV_8U);

	// Apply the Canny Edge Detection algorithm
	Canny(img.clone(), imgCanny, 200, 230);

	imshow("Original Image", img);
	imshow("Laplacian Edges", imgLaplacian);
	imshow("Canny Edges", imgCanny);

	imwrite("Laplacian_T40.jpg", imgLaplacian);
	imwrite("Canny_T1_200_T2_230.jpg", imgCanny);

	waitKey(0);
	return 0;
}