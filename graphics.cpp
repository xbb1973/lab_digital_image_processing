//
// Created by xbb1973 on 2020-03-03.
//

#include<iostream>
#include<string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

void lab1();

void binaryzation_image(string image_path, int channels_flag, double threshold);

void logarithm_transformation_image(string image_path, int channels_flag, double c);

void gama_transformation_image(string image_path, int channels_flag, double c, double gama);

void complementary_colour_transformation_image(string image_path);

const string image_path = "/Users/xbb1973/Documents/code/workdir/lab_digital_image/image/";

int main(){
    lab1();
    // Mat srcImage = imread(image_path + "1.jpeg", 1);
    // if (!srcImage.data) {
    //    std::cout << "Image not loaded";
    //    retur0n -1;
    // }
    // imshow("[img]", srcImage);
    // waitKey(0);
    return 0;
}

void lab1(){
    string image = image_path + "lena.png";
    binaryzation_image(image, 1, 100);
    logarithm_transformation_image(image, 0, 100);
    gama_transformation_image(image, 0, 100, 0.4);
    complementary_colour_transformation_image(image);
}

void binaryzation_image(string image_path, int channels_flag, double threshold) {
    Mat src_image = imread(image_path, channels_flag); //flag=0 灰度图单通道，flag>0 3通道
    for (int i = 0; i < src_image.rows; ++i) {
        for (int j = 0; j < src_image.cols; ++j) {
            if(channels_flag == 0){
                auto value = src_image.at<uchar>(i, j);
                if (value > threshold) value = 255;
                else value = 0;
                src_image.at<uchar>(i, j) = value;
            } else {
                auto value = src_image.at<Vec3b>(i, j);
                for(int i = 0; i < 3; i++){
                    if (value[i] > threshold) value[i] = 255;
                    else value = 0;
                }
                src_image.at<Vec3b>(i, j) = value;
            }
        }
    }
    imshow("binaryzation_image", src_image);
    waitKey(0);
    destroyAllWindows();
}

void logarithm_transformation_image(string image_path, int channels_flag, double c) {
    Mat src_image = imread(image_path, channels_flag); //flag=0 灰度图单通道，flag>0 3通道
//    s = c log (1 + r)
    for (int i = 0; i < src_image.rows; ++i) {
        for (int j = 0; j < src_image.cols; ++j) {
            if(channels_flag == 0){
                auto value = src_image.at<uchar>(i, j);
                src_image.at<uchar>(i, j) = uchar(c * log(double(value + 1)));
            } else {
                auto value = src_image.at<Vec3b>(i, j);
                for(int i = 0; i < 3; i++){
                    value[i] = uchar(c * log(double(value[i] + 1)));
                }
                src_image.at<Vec3b>(i, j) = value;
            }
        }
    }
    imshow("logarithm_transformation_image", src_image);
    waitKey(0);
    destroyAllWindows();
}

void gama_transformation_image(string image_path, int channels_flag, double c, double gama) {
    Mat src_image = imread(image_path, channels_flag); //flag=0 灰度图单通道，flag>0 3通道
//    s = c r^gama
    for (int i = 0; i < src_image.rows; ++i) {
        for (int j = 0; j < src_image.cols; ++j) {
            if(channels_flag == 0){
                auto value = src_image.at<uchar>(i, j);
                src_image.at<uchar>(i, j) = uchar(c * pow(value, gama));
            } else {
                auto value = src_image.at<Vec3b>(i, j);
                for(int i = 0; i < 3; i++){
                    value[i] = uchar(c * pow(value[i], gama));
                }
                src_image.at<Vec3b>(i, j) = value;
            }
        }
    }
    imshow("gama_transformation_image", src_image);
    waitKey(0);
    destroyAllWindows();
}

void complementary_colour_transformation_image(string image_path) {
    int channels_flag = 3;
    Mat src_image = imread(image_path, channels_flag); //flag=0 灰度图单通道，flag>0 3通道
//    s = c r^gama
//    pixel⾥的三个通道是BGR，其补⾊是CMY⾊域的，变换关系如下：
//    C=255-R； M=255-G； Y=255-B;
    for (int i = 0; i < src_image.rows; ++i) {
        for (int j = 0; j < src_image.cols; ++j) {
            if(channels_flag == 0){
//                do nothing;
            } else {
                auto value = src_image.at<Vec3b>(i, j);
                for(int i = 0; i < 3; i++){
                    value[i] = uchar(255 - value[2-i]);
                }
                src_image.at<Vec3b>(i, j) = value;
            }
        }
    }
    imshow("complementary_colour_transformation_image", src_image);
    waitKey(0);
    destroyAllWindows();
}
