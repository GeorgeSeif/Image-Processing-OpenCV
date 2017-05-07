#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

using namespace cv;

int main()
{
	Mat inputGaussian = imread("lena_gaussian_noise.png", CV_LOAD_IMAGE_UNCHANGED);
	Mat inputImpulse = imread("lena_impulse_noise.jpg", CV_LOAD_IMAGE_UNCHANGED);
	
	Mat imgBox3x3, imgBox7x7, imgGauss3x3, imgGauss7x7, imgMedian3x3, imgMedian7x7;

	namedWindow("Gaussian Noise Image", 1);
	namedWindow("Impulse Noise Image", 1);
	imshow("Gaussian Noise Image", inputGaussian);
	imshow("Impulse Noise Image", inputImpulse);

	boxFilter(inputGaussian.clone(), imgBox3x3, inputGaussian.depth(), Size(3, 3));
	boxFilter(inputGaussian.clone(), imgBox7x7, inputGaussian.depth(), Size(7, 7));

	int wndSize = 3;
	double sigma = (wndSize - 1) / 6;
	GaussianBlur(inputGaussian.clone(), imgGauss3x3, Size(wndSize, wndSize), sigma, sigma);
	wndSize = 7;
	sigma = (wndSize - 1) / 6;
	GaussianBlur(inputGaussian.clone(), imgGauss7x7, Size(wndSize, wndSize), sigma, sigma);

	medianBlur(inputGaussian.clone(), imgMedian3x3, 3);
	medianBlur(inputGaussian.clone(), imgMedian7x7, 7);

	//namedWindow("Gaussian Noise, 3x3 Box Blurred Image", 1);
	//namedWindow("Gaussian Noise, 3x3 Gaussian Blurred Image", 1);
	//namedWindow("Gaussian Noise, 3x3 Median Blurred Image", 1);
	//namedWindow("Gaussian Noise, 7x7 Box Blurred Image", 1);
	//namedWindow("Gaussian Noise, 7x7 Gaussian Blurred Image", 1);
	//namedWindow("Gaussian Noise, 7x7 Median Blurred Image", 1);

	//imshow("Gaussian Noise, 3x3 Box Blurred Image", imgBox3x3);
    //imshow("Gaussian Noise, 3x3 Gaussian Blurred Image", imgGauss3x3);
	//imshow("Gaussian Noise, 3x3 Median Blurred Image", imgMedian3x3);
	//imshow("Gaussian Noise, 7x7 Box Blurred Image", imgBox7x7);
	//imshow("Gaussian Noise, 7x7 Gaussian Blurred Image", imgGauss7x7);
	//imshow("Gaussian Noise, 7x7 Median Blurred Image", imgMedian7x7);

	imwrite("gaussian_noise_imgBox3x3.jpg", imgBox3x3);
	imwrite("gaussian_noise_imgBox7x7.jpg", imgBox7x7);
	imwrite("gaussian_noise_imgGauss3x3.jpg", imgGauss3x3);
	imwrite("gaussian_noise_imgGauss7x7.jpg", imgGauss7x7);
	imwrite("gaussian_noise_imgMedian3x3.jpg", imgMedian3x3);
	imwrite("gaussian_noise_imgMedian7x7.jpg", imgMedian7x7);

	boxFilter(inputImpulse.clone(), imgBox3x3, inputImpulse.depth(), Size(3, 3));
	boxFilter(inputImpulse.clone(), imgBox7x7, inputImpulse.depth(), Size(7, 7));

	wndSize = 3;
	sigma = (wndSize - 1) / 6;
	GaussianBlur(inputImpulse.clone(), imgGauss3x3, Size(wndSize, wndSize), sigma, sigma);
	wndSize = 7;
	sigma = (wndSize - 1) / 6;
	GaussianBlur(inputImpulse.clone(), imgGauss7x7, Size(wndSize, wndSize), sigma, sigma);

	medianBlur(inputImpulse.clone(), imgMedian3x3, 3);
	medianBlur(inputImpulse.clone(), imgMedian7x7, 7);

	namedWindow("Impulse Noise, 3x3 Box Blurred Image", 1);
	namedWindow("Impulse Noise, 3x3 Gaussian Blurred Image", 1);
	namedWindow("Impulse Noise, 3x3 Median Blurred Image", 1);
	namedWindow("Impulse Noise, 7x7 Box Blurred Image", 1);
	namedWindow("Impulse Noise, 7x7 Gaussian Blurred Image", 1);
	namedWindow("Impulse Noise, 7x7 Median Blurred Image", 1);

	imshow("Impulse Noise, 3x3 Box Blurred Image", imgBox3x3);
	imshow("Impulse Noise, 3x3 Gaussian Blurred Image", imgGauss3x3);
	imshow("Impulse Noise, 3x3 Median Blurred Image", imgMedian3x3);
	imshow("Impulse Noise, 7x7 Box Blurred Image", imgBox7x7);
	imshow("Impulse Noise, 7x7 Gaussian Blurred Image", imgGauss7x7);
	imshow("Impulse Noise, 7x7 Median Blurred Image", imgMedian7x7);

	imwrite("Impulse_noise_imgBox3x3.jpg", imgBox3x3);
	imwrite("Impulse_noise_imgBox7x7.jpg", imgBox7x7);
	imwrite("Impulse_noise_imgGauss3x3.jpg", imgGauss3x3);
	imwrite("Impulse_noise_imgGauss7x7.jpg", imgGauss7x7);
	imwrite("Impulse_noise_imgMedian3x3.jpg", imgMedian3x3);
	imwrite("Impulse_noise_imgMedian7x7.jpg", imgMedian7x7);

	waitKey(0);
	
	return 0;
}