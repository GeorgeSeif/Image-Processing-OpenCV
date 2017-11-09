#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>

#include <math.h> 
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;


// White balancing using the method in Gimp https://docs.gimp.org/en/gimp-layer-white-balance.html
Mat white_balance(Mat img, float perc = 0.05)
{
    CV_Assert(img.data);

    // Accept only char  colour type matrices
    CV_Assert(img.depth() != sizeof(uchar));
    // CV_Assert(img.channels() != 3);


    // Compute the histograms of the three colour channels
    int histSize = 256;

    // Set the range
    float range[] = { 0, 256 };
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat r_hist, g_hist, b_hist;

    Mat bgr[3];   //destination array
    split(img, bgr);//split source  

    // Compute the histogram
    calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

    Mat r_norm_hist = r_hist / bgr[2].total();
    Mat g_norm_hist = g_hist / bgr[1].total();
    Mat b_norm_hist = b_hist / bgr[0].total();
    



    // Now get the CDF

    std::vector<float> r_cdf;
    std::vector<float> g_cdf;
    std::vector<float> b_cdf;
    for (int i = 0; i < 256; i++)
    {
        if (i == 0)
        {
            r_cdf.push_back((r_norm_hist.at<float>(i, 0)));
            g_cdf.push_back((g_norm_hist.at<float>(i, 0)));
            b_cdf.push_back((b_norm_hist.at<float>(i, 0)));
        }
        else
        {
            r_cdf.push_back((r_norm_hist.at<float>(i, 0) + r_cdf[i-1]) );
            g_cdf.push_back((g_norm_hist.at<float>(i, 0) + g_cdf[i-1]) );
            b_cdf.push_back((b_norm_hist.at<float>(i, 0) + b_cdf[i-1]) );
        }
        
    }




    // Get the min and max pixel values indicating the 5% of pixels at the ends of the histograms
    float r_min_val = -1, g_min_val = -1, b_min_val = -1; // Dummy value
    float r_max_val = -1, g_max_val = -1, b_max_val = -1; // Dummy value

    for (int i = 0; i < 256; i++)
    {
        if (r_cdf[i] >= perc && r_min_val == -1)
        {
            r_min_val = (float)i;
        }

        if (r_cdf[i] >= (1.0 - perc) && r_max_val == -1)
        {
            cout << r_cdf[i] << endl;
            r_max_val = (float)i;
        }

        if (g_cdf[i] >= perc && g_min_val == -1)
        {
            g_min_val = (float)i;
        }

        if (g_cdf[i] >= (1.0 - perc) && g_max_val == -1)
        {
            g_max_val = (float)i;
        }

        if (b_cdf[i] >= perc && b_min_val == -1)
        {
            b_min_val = (float)i;
        }

        if (b_cdf[i] >= (1.0 - perc) && b_max_val == -1)
        {
            b_max_val = (float)i;
        }
    } 


    // Build look up table
    unsigned char r_lut[256], g_lut[256], b_lut[256];
    for (int i = 0; i < 256; i++)
    {   
        float index = (float)i;
        r_lut[i] = saturate_cast<uchar>(255.0 * ((index - r_min_val) / (r_max_val - r_min_val)));
        g_lut[i] = saturate_cast<uchar>(255.0 * ((index - g_min_val) / (g_max_val - g_min_val)));
        b_lut[i] = saturate_cast<uchar>(255.0 * ((index - b_min_val) / (b_max_val - b_min_val)));
    }



    Mat out = img.clone();
    MatIterator_<Vec3b> it, end;
    for (it = out.begin<Vec3b>(), end = out.end<Vec3b>(); it != end; it++)
    {

      (*it)[2] = r_lut[(*it)[2]];
      (*it)[1] = g_lut[(*it)[1]];
      (*it)[0] = b_lut[(*it)[0]];
    }

    
    return out;

}

float mean_pixel(Mat img)
{
    if (img.channels() > 2)
    {
        cvtColor(img.clone(), img, CV_BGR2GRAY);
        return mean(img)[0];
    }
    else
    {
        return mean(img)[0];
    }
}

float auto_gamma_value(Mat img)
{
    float max_pixel = 255;
    float middle_pixel = 128;
    float pixel_range = 256;
    float mean_l = mean_pixel(img);

    float gamma = log(middle_pixel/pixel_range)/ log(mean_l/pixel_range); // Formula from ImageJ

    return gamma;

}


Mat gamma_correction(Mat img, float gamma=0)
{
    CV_Assert(img.data);

    // Accept only char type matrices
    CV_Assert(img.depth() != sizeof(uchar));


    if (gamma == 0) {gamma = auto_gamma_value(img);}


    // Build look up table
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
    {
        lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
    }

    Mat out = img.clone();
    const int channels = out.channels();
    switch (channels)
    {
    case 1:
    {
              MatIterator_<uchar> it, end;
              for (it = out.begin<uchar>(), end = out.end<uchar>(); it != end; it++)
                  *it = lut[(*it)];

              break;
    }
    case 3:
    {
              MatIterator_<Vec3b> it, end;
              for (it = out.begin<Vec3b>(), end = out.end<Vec3b>(); it != end; it++)
              {

                  (*it)[0] = lut[((*it)[0])];
                  (*it)[1] = lut[((*it)[1])];
                  (*it)[2] = lut[((*it)[2])];
              }

              break;

    }
    }

    return out;
}


/*
    Huang, S. C., Cheng, F. C., & Chiu, Y. S. (2013). 
    Efficient contrast enhancement using adaptive gamma correction with weighting distribution. 
    IEEE Transactions on Image Processing, 22(3), 1032-1041.
*/
Mat adaptive_gamma_correction(Mat img, float alpha=0)
{

    CV_Assert(img.data);

    // Accept only char type matrices
    CV_Assert(img.depth() != sizeof(uchar));

    // Automatically compute the alpha value
    if (alpha == 0) {alpha = auto_gamma_value(img);}

    // Get the image probability density function using the histogram
    Mat gray_img;
    if (img.channels() > 2)
    {
        cvtColor(img.clone(), gray_img, CV_BGR2GRAY);
    }
    
    
    // Establish the number of bins
    int histSize = 256;

    // Set the range
    float range[] = { 0, 256 };
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat hist;

    // Compute the histogram
    calcHist(&gray_img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    Mat norm_hist = hist / gray_img.total();
    double pdf_min = 0;
    double pdf_max = 0;
    minMaxLoc(norm_hist, &pdf_min, &pdf_max);

    std::vector<float> pdf_weights;
    for (int i = 0; i < 256; i++)
    {
        pdf_weights.push_back(pdf_max * (pow(( norm_hist.at<float>(i, 0) - pdf_min) / (pdf_max - pdf_min), alpha)  ));
    }

    std::vector<float> cdf_weights;
    float pdf_weights_total = sum(pdf_weights)[0];
    for (int i = 0; i < 256; i++)
    {
        if (i == 0)
        {
            cdf_weights.push_back((pdf_weights[i]));
        }
        else
        {
            cdf_weights.push_back((pdf_weights[i] + cdf_weights[i-1]) );
        }
        
    }

    // Build look up table
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
    {   
        float gamma = 1 - cdf_weights[i];
        lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
    }

    Mat out = img.clone();
    const int num_channels = out.channels();
    switch (num_channels)
    {
    case 1:
    {
              MatIterator_<uchar> it, end;
              for (it = out.begin<uchar>(), end = out.end<uchar>(); it != end; it++)
                  *it = lut[(*it)];

              break;
    }
    case 3:
    {
              MatIterator_<Vec3b> it, end;
              for (it = out.begin<Vec3b>(), end = out.end<Vec3b>(); it != end; it++)
              {

                  (*it)[0] = lut[((*it)[0])];
                  (*it)[1] = lut[((*it)[1])];
                  (*it)[2] = lut[((*it)[2])];
              }

              break;

    }
    }

    return out;

}

int main()
{
    
    Mat img = imread("godfather.png", CV_LOAD_IMAGE_COLOR);

    std::cout << mean_pixel(img) << std::endl;


    Mat out;

    cout << "Denoising ..." << endl;
    fastNlMeansDenoisingColored(img, out);
    imwrite("denoised.png", out);

    // cout << "White balancing ..." << endl;
    // out = white_balance(out.clone());
    // imwrite("white_balanced.png", out);

    cout << "Adaptive gamma correction ..." << endl;
    out = adaptive_gamma_correction(out.clone(), 2);


    cout << "Saving ..." << endl;

    imwrite("out.png", out);

    return 0;
}