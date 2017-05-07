#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("parrot.jpg");
	Mat laplacianSharpening, unsharpSharpening;

    // Laplacian sharpening
	float laplacianBoostFactor = 1.2; // Controls the amount of Laplacian Sharpening
	Mat kern = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5*laplacianBoostFactor, -1, 0, -1, 0); // The filtering mask for Laplacian sharpening
	filter2D(img.clone(), laplacianSharpening, img.depth(), kern, Point(-1, -1)); // Applies the masking operator to the image

	// Unsharp mask sharpening
	Mat blur;
	GaussianBlur(img.clone(), blur, Size(5, 5), 0.67, 0.67);
	Mat unsharpMask = img.clone() - blur; // Compute the unsharp mask
	threshold(unsharpMask.clone(), unsharpMask, 20, 255, THRESH_TOZERO);
	float unsharpBoostFactor = 9; // Controls the amount of Unsharp Mask Sharpening
	unsharpSharpening = img.clone() + (unsharpMask * unsharpBoostFactor);

	imshow("Original Image", img);
	imshow("Laplacian Sharpened Image", laplacianSharpening);
	imshow("Unsharp Masking Sharpening", unsharpSharpening);

	imwrite("Laplacian_Sharp_Boost_1_point_2.jpg", laplacianSharpening);
	imwrite("Unsharp_Sharp_Boost_9_Threshold_20.jpg", unsharpSharpening);

	waitKey(0);
	return 0;
}