#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

int findBiggestContour(std::vector<std::vector<Point> > contours)
{
	int indexOfBiggestContour = -1;
	int sizeOfBiggestContour = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > sizeOfBiggestContour)
		{
			sizeOfBiggestContour = contours[i].size();
			indexOfBiggestContour = i;
		}
	}
	return indexOfBiggestContour;
}

int main()
{
	Mat img = imread("hand.png");
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", img);

	// First smooth the image 
	GaussianBlur(img.clone(), img, Size(5, 5), 0.67, 0.67);
	
	// Convert the image to the HSV colour space for segmentation processing
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);

	// Segment the image based on the specified colour range 
	Mat segmented_img;
	inRange(hsv, Scalar(0, 10, 60), Scalar(20, 150, 255), segmented_img);

	// ******OPTIONAL*******
	// Select only the biggest item within the colour range
	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	findContours(segmented_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	int s = findBiggestContour(contours);

	Mat drawing = Mat::zeros(img.size(), CV_8UC1);
	drawContours(drawing, contours, s, Scalar(255), -1, 8, hierarchy, 0, Point());

	// Create a structuring element
	int dialation_size = 6;
	Mat element = getStructuringElement(cv::MORPH_CROSS,
		cv::Size(2 * dialation_size + 1, 2 * dialation_size + 1),
		cv::Point(dialation_size, dialation_size));

	// Apply dilation on the image
	dilate(drawing.clone(), drawing, element);  // dilate(image,dst,element);

	namedWindow("Segmented Image", WINDOW_AUTOSIZE);
	imshow("Segmented Image", drawing);

	imwrite("drawing.jpg", drawing);

	waitKey(0);

	return 0;
}
