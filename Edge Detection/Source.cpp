#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("tulips.png", CV_LOAD_IMAGE_UNCHANGED);
	Mat blur, gray, GrayEdges, finalColourEdges;
	vector<Mat> rgbColourEdges(3);

	// Smooth the image to reduce noise for more accurate edge detection
	GaussianBlur(img.clone(), blur, Size(3, 3), 0, 0, BORDER_DEFAULT);

	// Convert it to grayscale
	cvtColor(blur.clone(), gray, CV_BGR2GRAY);

	// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	// Gradient X
	//Scharr(gray, grad_x, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT);
	Sobel(gray, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	// Gradient Y
	//Scharr(gray, grad_y, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT);
	Sobel(gray, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	// Calculate the magnitude
	Mat temp, tempFloat;
	pow(grad_x, 2, grad_x);
	pow(grad_y, 2, grad_y);
	temp = grad_x + grad_y;
	
	temp.convertTo(tempFloat, CV_32F); // We need CV_32F for the sqrt() function
	sqrt(temp, GrayEdges);

	GrayEdges.convertTo(GrayEdges, CV_8U); // Convert back for display

	double T = 120;
	double maxval = 255;
	threshold(GrayEdges, GrayEdges, T, maxval, THRESH_BINARY);

	imshow("Original Image", img);
	imshow("Grayscale Edge Detection", GrayEdges);
	//imwrite("Scharr_GrayEdges_thresh120.jpg", GrayEdges);

	// Now for colour edge detection
	vector<Mat> channels(3);
	split(blur.clone(), channels);

	for (int i = 0; i < 3; i++)
	{
		// Generate grad_x and grad_y
		Mat colour_grad_x, colour_grad_y;
		Mat colour_abs_grad_x, colour_abs_grad_y;

		// Gradient X
		//Scharr(channels[i], colour_grad_x, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT);
		Sobel(channels[i], colour_grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(colour_grad_x, colour_abs_grad_x);

		// Gradient Y
		//Scharr(channels[i], colour_grad_y, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT);
		Sobel(channels[i], colour_grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(colour_grad_y, colour_abs_grad_y);

		// Calculate the magnitude
		Mat colour_temp, colour_tempFloat;
		pow(colour_grad_x, 2, colour_grad_x);
		pow(colour_grad_y, 2, colour_grad_y);
		colour_temp = colour_grad_x + colour_grad_y;

		colour_temp.convertTo(colour_tempFloat, CV_32F); // We need CV_32F for the sqrt() function
		sqrt(colour_temp, rgbColourEdges[i]);

		threshold(rgbColourEdges[i], rgbColourEdges[i], T, maxval, THRESH_BINARY);
	}

	Mat checkEdges;
	bitwise_or(rgbColourEdges[0], rgbColourEdges[1], checkEdges);
	bitwise_or(checkEdges, rgbColourEdges[2], finalColourEdges);

	finalColourEdges.convertTo(finalColourEdges, CV_8U);

	imshow("Colour Edge Detection", finalColourEdges);
	//imwrite("Scharr_ColourEdges_thresh120_OR.jpg", finalColourEdges);

	waitKey(0);
	return 0;
}