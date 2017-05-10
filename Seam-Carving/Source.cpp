#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <limits>
#include <iostream>

#include "cpp_util.hpp"

using namespace cv;
using std::cout;
using std::cin;
using std::endl;

// Compute the energy matrix of the image
Mat compute_energy_matrix(Mat img)
{

	// For colour edge detection
	std::vector<Mat> channels(3);
	split(img.clone(), channels);
	std::vector<Mat> rgbColourEdges(3);

	for (int i = 0; i < 3; i++)
	{
		// Generate grad_x and grad_y
		Mat colour_grad_x, colour_grad_y;

		// Gradient X
		//Scharr(channels[i], colour_grad_x, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT);
		Sobel(channels[i], colour_grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);

		// Gradient Y
		//Scharr(channels[i], colour_grad_y, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT);
		Sobel(channels[i], colour_grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);

		rgbColourEdges[i] = abs(colour_grad_x) + abs(colour_grad_y);
		
		// Convert to float for the energy function
		rgbColourEdges[i].clone().convertTo(rgbColourEdges[i], CV_32F);

	}

	// Compute the final energy function by summing all of the colour channels
	Mat energy_matrix = Mat::zeros(img.size(), CV_32F);
	add(rgbColourEdges[0], rgbColourEdges[1], energy_matrix);
	add(rgbColourEdges[2], energy_matrix.clone(), energy_matrix);

	return energy_matrix;

}

// Compute the score matrix using dynamic programming, given the energy function
Mat compute_score_matrix(Mat energy_matrix)
{
	Mat score_matrix = Mat::zeros(energy_matrix.size(), CV_32F);
	score_matrix.row(0) = energy_matrix.row(0);

	for (int i = 1; i < score_matrix.rows; i++)
	{
		for (int j = 0; j < score_matrix.cols; j++)
		{
			float min_score = 0;

			// Handle the edge cases
			if (j - 1 < 0)
			{
				std::vector<float> scores(2);
				scores[0] = score_matrix.at<float>(i - 1, j);
				scores[1] = score_matrix.at<float>(i - 1, j + 1);
				min_score = *std::min_element(std::begin(scores), std::end(scores));
			}
			else if (j + 1 >= score_matrix.cols)
			{
				std::vector<float> scores(2);
				scores[0] = score_matrix.at<float>(i - 1, j - 1);
				scores[1] = score_matrix.at<float>(i - 1, j);
				min_score = *std::min_element(std::begin(scores), std::end(scores));
			}	
			else
			{
				std::vector<float> scores(3);
				scores[0] = score_matrix.at<float>(i - 1, j - 1);
				scores[1] = score_matrix.at<float>(i - 1, j);
				scores[2] = score_matrix.at<float>(i - 1, j + 1);
				min_score = *std::min_element(std::begin(scores), std::end(scores));
			}
						
			score_matrix.at<float>(i, j) = energy_matrix.at<float>(i, j) + min_score;
		}
	}

	return score_matrix;
}

// Given the score matrix, traverse back up to get the lowest energy seam
std::vector<int> get_seam(Mat score_matrix)
{
	// Get the lowest score in the last row
	std::vector<float> last_row = score_matrix.row(score_matrix.rows - 1);
	int end_idx = std::min_element(std::begin(last_row), std::end(last_row)) - std::begin(last_row);

	// Extract the lowest energy seam
	std::vector<int> low_energy_seam(score_matrix.rows);
	low_energy_seam[low_energy_seam.size() - 1] = end_idx;
	int curr_col_idx = end_idx;
	float inf = std::numeric_limits<float>::infinity();

	for (int i = score_matrix.rows - 1; i > 0; i--)
	{
		int left_col_idx = curr_col_idx - 1;
		int right_col_idx = curr_col_idx + 1;
		int	next_idx = 1;
		std::vector<float> scores(3);

		if (left_col_idx < 0 || left_col_idx >= score_matrix.cols)
		{
			scores = { inf, score_matrix.at<float>(i, curr_col_idx), score_matrix.at<float>(i, right_col_idx) };
		}
		else if (right_col_idx < 0 || right_col_idx >= score_matrix.cols)
		{
			scores = { score_matrix.at<float>(i, left_col_idx), score_matrix.at<float>(i, curr_col_idx), inf };
		}
		else
		{
			scores = { score_matrix.at<float>(i, left_col_idx), score_matrix.at<float>(i, curr_col_idx), score_matrix.at<float>(i, right_col_idx) };
		}

			next_idx = std::min_element(std::begin(scores), std::end(scores)) - std::begin(scores);

			if (next_idx == 0)
			{
				curr_col_idx = left_col_idx;
			}
		
			if (next_idx == 3)
			{
				curr_col_idx = right_col_idx;
			}

			low_energy_seam[i - 1] = curr_col_idx;
	}


	return low_energy_seam;

}

Mat remove_single_seam(Mat img, std::vector<int> low_energy_seam)
{
	Mat out = Mat::zeros(Size(img.cols - 1, img.rows), img.type());
	int ch = img.channels();

	for (int i = 0; i < img.rows; i++)
	{
		int out_col_idx = 0;

		// Create a pointer to the pixel array
		uchar *img_p = img.ptr<uchar>(i);
		uchar *out_p = out.ptr<uchar>(i);
		for (int j = 0; j < img.cols; j++)
		{
			bool keep_pixel = true;

			if (low_energy_seam.at(i) == j)
			{
				keep_pixel = false;
			}

			if (keep_pixel == true)
			{
				out_p[out_col_idx*ch + 0] = img_p[j*ch + 0]; // B
				out_p[out_col_idx*ch + 1] = img_p[j*ch + 1]; // G
				out_p[out_col_idx*ch + 2] = img_p[j*ch + 2]; // R
				out_col_idx += 1;
			}
			else 
			{
				continue;
			}
		}
	}

	return out;
}

Mat add_single_seam(Mat img, std::vector<int> low_energy_seam)
{
	Mat out = Mat::zeros(Size(img.cols + 1, img.rows), img.type());
	int ch = img.channels();

	for (int i = 0; i < img.rows; i++)
	{
		int img_col_idx = 0;

		// Create a pointer to the pixel array
		uchar *img_p = img.ptr<uchar>(i);
		uchar *out_p = out.ptr<uchar>(i);
		for (int j = 0; j < img.cols + 1; j++)
		{
			bool new_pixel = false;

			if (low_energy_seam.at(i) == j)
			{
				new_pixel = true;
			}

			if (new_pixel == false)
			{
				out_p[j*ch + 0] = img_p[img_col_idx*ch + 0]; // B
				out_p[j*ch + 1] = img_p[img_col_idx*ch + 1]; // G
				out_p[j*ch + 2] = img_p[img_col_idx*ch + 2]; // R
				img_col_idx += 1;
			}
			else
			{
				// Handle the edge cases
				if (img_col_idx - 1 < 0)
				{
					out_p[j*ch + 0] = (img_p[img_col_idx*ch + 0] + img_p[(img_col_idx + 1)*ch + 0]) / 2; // B
					out_p[j*ch + 1] = (img_p[img_col_idx*ch + 1] + img_p[(img_col_idx + 1)*ch + 1]) / 2; // G
					out_p[j*ch + 2] = (img_p[img_col_idx*ch + 2] + img_p[(img_col_idx + 1)*ch + 2]) / 2; // R
				}
				else if (img_col_idx + 1 >= img.cols)
				{
					out_p[j*ch + 0] = (img_p[(img_col_idx - 1)*ch + 0] + img_p[img_col_idx*ch + 0]) / 2; // B
					out_p[j*ch + 1] = (img_p[(img_col_idx - 1)*ch + 1] + img_p[img_col_idx*ch + 1]) / 2; // G
					out_p[j*ch + 2] = (img_p[(img_col_idx - 1)*ch + 2] + img_p[img_col_idx*ch + 2]) / 2; // R
				}
				else
				{
					out_p[j*ch + 0] = (img_p[(img_col_idx - 1)*ch + 0] + img_p[img_col_idx*ch + 0] + img_p[(img_col_idx + 1)*ch + 0]) / 3; // B
					out_p[j*ch + 1] = (img_p[(img_col_idx - 1)*ch + 1] + img_p[img_col_idx*ch + 1] + img_p[(img_col_idx + 1)*ch + 1]) / 3; // G
					out_p[j*ch + 2] = (img_p[(img_col_idx - 1)*ch + 2] + img_p[img_col_idx*ch + 2] + img_p[(img_col_idx + 1)*ch + 2]) / 3; // R
				}
				
				img_col_idx = img_col_idx;
			}
		}
	}

	return out;

}

Mat remove_all_seams(Mat img, int num_seams)
{
	for (int i = 1; i <= num_seams; i++)
	{
		Mat energy = compute_energy_matrix(img.clone());
		Mat score = compute_score_matrix(energy);
		std::vector<int> seam = get_seam(score);
		cout << "Removing col seam # " << i << endl;
		img = remove_single_seam(img, seam);
	}

	return img;
}

Mat add_all_seams(Mat img, int num_seams)
{
	// Getting all of the seams to duplicate and add
	// These are the first "num_seams" that we would normally remove
	Mat img_copy = img.clone();

	cout << "Getting all seams " << num_seams << " to add..." << endl;
	std::vector<std::vector<int>> seams_to_add(num_seams);
	for (int i = 0; i < num_seams; i++)
	{
		Mat energy = compute_energy_matrix(img_copy.clone());
		Mat score = compute_score_matrix(energy);
		std::vector<int> seam = get_seam(score);
		seams_to_add.at(i) = seam;
		img_copy = remove_single_seam(img_copy, seam);
		cout << "Got seam # " << i + 1 << endl;
	}

	// Add all of the seams
	for (int i = 0; i < num_seams; i++)
	{
		cout << "Adding seam # " << i+1 << endl;
		img = add_single_seam(img, seams_to_add.at(i));
	}

	return img;
	
}


Mat seam_carving(Mat img, Size out_size)
{
	int num_row_seams = img.rows - out_size.height;
	int num_col_seams = img.cols - out_size.width;

	if (num_col_seams < 0)
	{
		num_col_seams = std::abs(num_col_seams);
		img = add_all_seams(img, num_col_seams);
	}
	else
	{
		img = remove_all_seams(img, num_col_seams);
	}

	transpose(img, img);

	if (num_row_seams < 0)
	{
		num_row_seams = std::abs(num_row_seams);
		img = add_all_seams(img, num_row_seams);
	}
	else
	{
		img = remove_all_seams(img, num_row_seams);
	}

	transpose(img, img);

	return img;
}

int main()
{
	// Input image
	Mat img = imread("ryerson.jpg");
	Mat img_copy = img.clone();

	if (img.empty())
	{
		cout << "ERROR: Image is empty!!!" << endl;
		return 0;
	}

	// Prompt the user to input the image size
	cout << "The image is of size = " << img.cols << " x " << img.rows << ". Enter in the new image size." << endl << endl;

	int num_rows = 0; 
	int num_cols = 0;
	cout << "rows: "; cin >> num_rows;
	cout << "cols: "; cin >> num_cols;
	cout << endl;

	// Get the number of seams to add or remove

	// Add or remove seams according to the difference in number of rows and cols
	Mat out = seam_carving(img, Size(num_cols, num_rows));

	namedWindow("Input Image", 0);
	namedWindow("Output Image", 0);
	imshow("Input Image", img_copy);
	imshow("Output Image", out);

	waitKey(0);

	return 0;
}