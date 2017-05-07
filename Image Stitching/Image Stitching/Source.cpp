#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\core.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\xfeatures2d.hpp>

using namespace cv;

Mat crop_black_borders(Mat img)
{
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	Mat binary_img;
	threshold(gray, binary_img, 1, 255, CV_THRESH_BINARY);

	std::vector<std::vector<Point> > contours;
	findContours(binary_img, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	std::vector<std::vector<Point> > contours_poly(contours.size());
	approxPolyDP(Mat(contours[0]), contours_poly[0], 3, true);
	Rect boundRect = boundingRect(Mat(contours[0]));
	Mat out = img(boundRect);
	
	return out;
}

int main()
{
	Mat img_1 = imread("photos/Ryerson-left.jpg");
	Mat img_2 = imread("photos/Ryerson-right.jpg");	

	// Detecting keypoints and compute descriptors
	Ptr<ORB> orb = ORB::create();
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	orb->detectAndCompute(img_1, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(img_2, Mat(), keypoints2, descriptors2);

	// Matching descriptors
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming");
	std::vector<DMatch> matches;
	descriptorMatcher->match(descriptors1, descriptors2, matches, Mat());

	// Drawing the results
	// Keep best matches only to have a nice drawing.
	// We sort distance between descriptor matches
	Mat index;
	int nbMatch = int(matches.size());
	Mat tab(nbMatch, 1, CV_32F);
	for (int i = 0; i<nbMatch; i++)
	{
		tab.at<float>(i, 0) = matches[i].distance;
	}
	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
	std::vector<DMatch> bestMatches;
	for (int i = 0; i < 200; i++)
	{
		bestMatches.push_back(matches[index.at<int>(i, 0)]);
	}
	Mat matching_result;
	drawMatches(img_1, keypoints1, img_2, keypoints2, bestMatches, matching_result);


	// Localize the object
	std::vector<Point2f> img_1_points;
	std::vector<Point2f> img_2_points;
	for (size_t i = 0; i < bestMatches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		img_1_points.push_back(keypoints1[bestMatches[i].queryIdx].pt);
		img_2_points.push_back(keypoints2[bestMatches[i].trainIdx].pt);
	}
	Mat H = findHomography(img_1_points, img_2_points, RANSAC);

	Mat stitching_result;
	warpPerspective(img_1, stitching_result, H, Size(img_1.cols + img_2.cols, img_1.rows));
	Mat half(stitching_result, Rect(0, 0, img_2.cols, img_2.rows));
	img_2.copyTo(half);

	Mat cropped_stitching_result = crop_black_borders(stitching_result);

	namedWindow("Result", 0);
	imshow("Result", cropped_stitching_result);

	waitKey(0);

	return 0;
}