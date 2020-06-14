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

#define GRAY_LEVEL 8
#define WIDTH 10
#define HEIGHT 10
#define M_WIDTH 5
#define M_HEIGHT 5
#define M_PI 3.14159265358979323846

// 本地测试图片路径
const string image_path = "/Users/xbb1973/Documents/code/workdir/lab_digital_image/image/";

// lab1负责调用实验1的内容逻辑
void lab1();

// 2、灰度图像二值化处理 具体内容：设置并调整阈值对图像进行二值化处理。
void binaryzation_image(string image_path, int channels_flag, double threshold);

// 3、灰度图像的对数变换 具体内容：设置并调整 r 值对图像进行对数变换。
void logarithm_transformation_image(string image_path, int channels_flag, double c);

// 4、灰度图像的伽马变换 具体内容：设置并调整γ值对图像进行伽马变换。
void gama_transformation_image(string image_path, int channels_flag, double c, double gama);

// 5、彩色图像的补色变换 具体内容：对彩色图像进行补色变换。
void complementary_colour_transformation_image(string image_path);

// lab2负责调用实验2的内容逻辑
void lab2();

Mat my_calcHist(Mat *images, int nimages, int channels, int dims, int *histSize, const float **ranges);

Mat my_normalize(Mat src, Mat dst, int alpha, int beta, NormTypes types);

Mat myEqualizeHist(Mat src, Mat dst);

// 1、计算灰度图像的归一化直方图。具体内容：利用 OpenCV 对图像像素进行操作，计算归一化直方图.并在窗口中以图形的方式显示出来
void normalnization_hist_gray_image(string image_path);

void normalnization_hist_color_image(string image_path);

// 2、灰度图像直方图均衡处理 具体内容：通过计算归一化直方图,设计算法实现直方图均衡化处理。
void normalnization_equalization_hist_gray_image(string image_path);

// 3、彩色图像直方图均衡处理 具体内容：在灰度图像直方图均衡处理的基础上实现彩色直方图均衡处理。
void normalnization_equalization_hist_color_image(string image_path, int flag);


/**
 * lab3负责调用实验3的内容逻辑
  * 1、利用均值模板平滑灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 3*3、5*5 和 9*9 尺寸的均值模板平滑灰度图像
  * 2、利用高斯模板平滑灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 3*3、5*5 和 9*9 尺寸的高斯模板平滑灰度图像
  * 3、利用 Laplacian、Robert、Sobel 模板锐化灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 Laplacian、Robert、 Sobel 模板锐化灰度图像
  * 4、利用高提升滤波算法增强灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，设计高提升滤波算法增 强图像
  * 5、利用均值模板平滑彩色图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，利 用 3*3、5*5 和 9*9 尺寸的均值模板平滑彩色图像
  * 6、利用高斯模板平滑彩色图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，分 别利用 3*3、5*5 和 9*9 尺寸的高斯模板平滑彩色图像
  * 7、利用 Laplacian、Robert、Sobel 模板锐化灰度图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，分 别利用 Laplacian、Robert、Sobel 模板锐化彩色图像
  *
  */
void lab3();

// 根据mask模板，对图像进行卷积，此方法不对图像进行Padding填充，因此会缩小图像
void CovImage(Mat src, Mat &dst, double *mask, int m_width, int m_height);

// 根据mask模板，对图像进行卷积，然后将原图像与锐化图像按比例混合，用于实现高提升滤波算法增强灰度图像
void EnhanceCovImage(Mat src, Mat &dst, double *mask, int m_width, int m_height);

// 生成width * height的均值模板，均值模板公式：1.0 / (width * height);
void MeanMask(double *mask, int width, int height);

// 生成width * height的高斯模板，高斯模板公式：G(x,y) = e ^ (-(x^2+y^2) / 2 * sigma^2)
void GaussianMask(double *mask, int width, int height, double sigma);

// Sobel滤波器模板生成，目前固定生成3*3的固定Sobel算子
void SobelMask(double *mask, int width, int height);

// Robert滤波器模板生成，目前固定生成3*3的固定Robert算子
void RobertMask(double *mask, int width, int height);

// Laplace滤波器模板生成，目前固定生成3*3的固定Laplace算子
void LaplaceMask(double *mask, int width, int height);

// 利用均值模板平滑灰度图像/彩色图像
Mat MeanFilter(string image_path, int m_width, int m_height, int flag);

// 利用高斯模板平滑灰度图像/彩色图像
Mat GaussianFilter(string image_path, int m_width, int m_height, double sigma, int flag);

// 利用Sobel算子锐化灰度图像/彩色图像
Mat SobelFilter(string image_path, int m_width, int m_height, int flag);

// 利用Robert算子锐化灰度图像/彩色图像
Mat RobertFilter(string image_path, int m_width, int m_height, int flag);

// 利用Laplace算子锐化灰度图像/彩色图像
Mat LaplaceFilter(string image_path, int m_width, int m_height, int flag);

// 利用高提升滤波算法增强灰度图像/彩色图像，其基本原理与上述一致。
// 唯一的改变就是调用了EnhanceCovImage对图像进行卷积后将原图像与锐化图像按比例混合，用于实现高提升滤波算法增强灰度图像
Mat EnhanceSobelFilter(string image_path, int m_width, int m_height, int flag);

Mat EnhanceRobertFilter(string image_path, int m_width, int m_height, int flag);

Mat EnhanceLaplaceFilter(string image_path, int m_width, int m_height, int flag);

/**
 * lab4
 * 实验 4：图像去噪
 * 1、均值滤波 具体内容：利用 OpenCV 对灰度图像像素进行操作，分别利用算术均值滤 波器、几何均值滤波器、谐波和逆谐波均值滤波器进行图像去噪。模板大小为 5*5。（注：请分别为图像添加高斯噪声、胡椒噪声、盐噪声和椒盐噪声，并观察 滤波效果）
 * 2、中值滤波 具体内容：利用 OpenCV 对灰度图像像素进行操作，分别利用 5*5 和 9*9 尺寸的模板对图像进行中值滤波。（注：请分别为图像添加胡椒噪声、盐噪声和 椒盐噪声，并观察滤波效果）
 * 3、自适应均值滤波。 具体内容：利用 OpenCV 对灰度图像像素进行操作，设计自适应局部降 低噪声滤波器去噪算法。模板大小 7*7（对比该算法的效果和均值滤波器的效果）
 * 4、自适应中值滤波 具体内容：利用 OpenCV 对灰度图像像素进行操作，设计自适应中值滤波算 法对椒盐图像进行去噪。模板大小 7*7（对比中值滤波器的效果）
 * 5、彩色图像均值滤波 具体内容：利用 OpenCV 对彩色图像 RGB 三个通道的像素进行操作，利用算 术均值滤波器和几何均值滤波器进行彩色图像去噪。模板大小为 5*5。
 */
void lab4();

int main() {
    // lab1();

    // 此处用于测试图片读取路径、逻辑是否正确。
    // lab1
    // 1、利用 OpenCV 读取图像。 具体内容：用打开 OpenCV 打开图像，并在窗口中显示
    // Mat srcImage = imread(image_path + "1.jpeg", 1);
    // if (!srcImage.data) {
    //    std::cout << "Image not loaded";
    //    retur0n -1;
    // }
    // imshow("[img]", srcImage);
    // waitKey(0);

    // lab2();

    // lab3();

    lab4();

    return 0;
}

/**
 * 实验 1：图像灰度变换
 * 1、利用 OpenCV 读取图像。 具体内容：用打开 OpenCV 打开图像，并在窗口中显示
 * 2、灰度图像二值化处理 具体内容：设置并调整阈值对图像进行二值化处理。
 * 3、灰度图像的对数变换 具体内容：设置并调整 r 值对图像进行对数变换。
 * 4、灰度图像的伽马变换 具体内容：设置并调整γ值对图像进行伽马变换。
 * 5、彩色图像的补色变换 具体内容：对彩色图像进行补色变换。
 */
void lab1() {
    string image = image_path + "lena.png";
    // 2、灰度图像二值化处理 具体内容：设置并调整阈值对图像进行二值化处理。
    binaryzation_image(image, 1, 100);
    // 3、灰度图像的对数变换 具体内容：设置并调整 r 值对图像进行对数变换。
    logarithm_transformation_image(image, 0, 100);
    // 4、灰度图像的伽马变换 具体内容：设置并调整γ值对图像进行伽马变换。
    gama_transformation_image(image, 0, 100, 0.4);
    // 5、彩色图像的补色变换 具体内容：对彩色图像进行补色变换。
    complementary_colour_transformation_image(image);
}

// 2、灰度图像二值化处理 具体内容：设置并调整阈值对图像进行二值化处理。
void binaryzation_image(string image_path, int channels_flag, double threshold) {
    Mat src_image = imread(image_path, channels_flag); //flag=0 灰度图单通道，flag>0 3通道
    for (int i = 0; i < src_image.rows; ++i) {
        for (int j = 0; j < src_image.cols; ++j) {
            if (channels_flag == 0) {
                auto value = src_image.at<uchar>(i, j);
                if (value > threshold) value = 255;
                else value = 0;
                src_image.at<uchar>(i, j) = value;
            } else {
                auto value = src_image.at<Vec3b>(i, j);
                for (int i = 0; i < 3; i++) {
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

// 3、灰度图像的对数变换 具体内容：设置并调整 r 值对图像进行对数变换。
void logarithm_transformation_image(string image_path, int channels_flag, double c) {
    Mat src_image = imread(image_path, channels_flag); //flag=0 灰度图单通道，flag>0 3通道
//    s = c log (1 + r)
    for (int i = 0; i < src_image.rows; ++i) {
        for (int j = 0; j < src_image.cols; ++j) {
            if (channels_flag == 0) {
                auto value = src_image.at<uchar>(i, j);
                src_image.at<uchar>(i, j) = uchar(c * log(double(value + 1)));
            } else {
                auto value = src_image.at<Vec3b>(i, j);
                for (int i = 0; i < 3; i++) {
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

// 4、灰度图像的伽马变换 具体内容：设置并调整γ值对图像进行伽马变换。
void gama_transformation_image(string image_path, int channels_flag, double c, double gama) {
    Mat src_image = imread(image_path, channels_flag); //flag=0 灰度图单通道，flag>0 3通道
//    s = c r^gama
    for (int i = 0; i < src_image.rows; ++i) {
        for (int j = 0; j < src_image.cols; ++j) {
            if (channels_flag == 0) {
                auto value = src_image.at<uchar>(i, j);
                src_image.at<uchar>(i, j) = uchar(c * pow(value, gama));
            } else {
                auto value = src_image.at<Vec3b>(i, j);
                for (int i = 0; i < 3; i++) {
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

// 5、彩色图像的补色变换 具体内容：对彩色图像进行补色变换。
void complementary_colour_transformation_image(string image_path) {
    int channels_flag = 3;
    Mat src_image = imread(image_path, channels_flag); //flag=0 灰度图单通道，flag>0 3通道
//    s = c r^gama
//    pixel⾥的三个通道是BGR，其补⾊是CMY⾊域的，变换关系如下：
//    C=255-R； M=255-G； Y=255-B;
    for (int i = 0; i < src_image.rows; ++i) {
        for (int j = 0; j < src_image.cols; ++j) {
            if (channels_flag == 0) {
                // do nothing;
            } else {
                auto value = src_image.at<Vec3b>(i, j);
                for (int i = 0; i < 3; i++) {
                    value[i] = uchar(255 - value[2 - i]);
                }
                src_image.at<Vec3b>(i, j) = value;
            }
        }
    }
    imshow("complementary_colour_transformation_image", src_image);
    waitKey(0);
    destroyAllWindows();
}

/**
 * 1、计算灰度图像的归一化直方图。具体内容：利用 OpenCV 对图像像素进行操作，计算归一化直方图.并在窗口中以图形的方式显示出来
 * 2、灰度图像直方图均衡处理 具体内容：通过计算归一化直方图,设计算法实现直方图均衡化处理。
 * 3、彩色图像直方图均衡处理 具体内容：在灰度图像直方图均衡处理的基础上实现彩色直方图均衡处理。
 */
void lab2() {
    string image = image_path + "lena.png";
    // 1、计算灰度图像的归一化直方图。具体内容：利用 OpenCV 对图像像素进行操作，计算归一化直方图.并在窗口中以图形的方式显示出来
    // normalnization_hist_color_image(image);
    normalnization_hist_gray_image(image);
    // 2、灰度图像直方图均衡处理 具体内容：通过计算归一化直方图,设计算法实现直方图均衡化处理。
    // normalnization_equalization_hist_gray_image(image);
    normalnization_equalization_hist_color_image(image, 0);
    // 3、彩色图像直方图均衡处理 具体内容：在灰度图像直方图均衡处理的基础上实现彩色直方图均衡处理。
    normalnization_equalization_hist_color_image(image, 1);
}

void normalnization_hist_color_image(string image_path) {
    // 读入图像
    // Imread flags
    //     enum ImreadModes {
    //         IMREAD_UNCHANGED            = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
    //         IMREAD_GRAYSCALE            = 0,  //!< If set, always convert image to the single channel grayscale image (codec internal conversion).
    //         IMREAD_COLOR                = 1,  //!< If set, always convert image to the 3 channel BGR color image.
    Mat srcImage = imread(image_path, 1);
    if (!srcImage.data) {
        printf("读取图片失败");
    }
    imshow("原图", srcImage);
    waitKey(0);

    // 1、计算直方图
    // 定义直方图的各个参数变量
    int numbins = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    // 将原图分割为BGR三个维度
    vector<Mat> bgr;
    split(srcImage, bgr);
    // 调用calcHist计算/统计
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbins, &histRange);
    calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &numbins, &histRange);
    calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &numbins, &histRange);

    // 2、绘制直方图
    // 定义直方图的参数
    int width = 512;//定义直方图的宽度
    int height = 300;//定义直方图的长度
    // 直方图白板
    Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));
    // 归一化直方图的高度
    // enum NormTypes   NORM_MINMAX    = 32 //!< flag
    normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
    normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
    normalize(r_hist, r_hist, 0, height, NORM_MINMAX);
    // 使用line绘制直方图
    int binStep = cvRound((float) width / (float) numbins);
    for (int i = 1; i < numbins; i++) {
        line(histImage,
             Point(binStep * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),
             Point(binStep * (i), height - cvRound(b_hist.at<float>(i))),
             Scalar(255, 0, 0));
        line(histImage,
             Point(binStep * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
             Point(binStep * (i), height - cvRound(g_hist.at<float>(i))),
             Scalar(0, 255, 0));
        line(histImage,
             Point(binStep * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
             Point(binStep * (i), height - cvRound(r_hist.at<float>(i))),
             Scalar(0, 0, 255));
    }
    imshow("Histogram", histImage);
    waitKey(0);
}

void normalnization_hist_gray_image(string image_path) {
    // 读入图像
    // Imread flags
    //     enum ImreadModes {
    //         IMREAD_UNCHANGED            = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
    //         IMREAD_GRAYSCALE            = 0,  //!< If set, always convert image to the single channel grayscale image (codec internal conversion).
    //         IMREAD_COLOR                = 1,  //!< If set, always convert image to the 3 channel BGR color image.
    Mat srcImage = imread(image_path, 0);
    if (!srcImage.data) {
        printf("读取图片失败");
    }
    imshow("原图", srcImage);
    // waitKey(0);

    // 1、计算直方图
    // 定义直方图的各个参数变量
    int numbins = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    // 将原图分割为BGR三个维度
    vector<Mat> bgr;
    split(srcImage, bgr);
    // 调用calcHist计算/统计
    // Mat b_hist, g_hist, r_hist;
    // calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbins, &histRange);
    // 手写calcHist
    Mat b_hist = my_calcHist(&bgr[0], 1, 0, 1, &numbins, &histRange);

    // 测试使用代码
    if (false) {
        // // Mat::Mat(int _rows, int _cols, int _type)
        // Mat b_hist(numbins, 1, CV_64FC1, Scalar(0));
        // // int offset = 256 / numbins;
        // printf("%d, %d \n", bgr[0].rows, bgr[0].cols);
        // for (int j = 0; j < bgr[0].rows; ++j) {
        //     for (int i = 0; i < bgr[0].cols; ++i) {
        //         // int index = bgr[0].at<uchar>(j, i) / offset;
        //         int index = bgr[0].at<uchar>(j, i);
        //         b_hist.at<float>(index, 0)++;
        //     }
        // }
        //
        // float max = 6460, min = 0;
        // float offset = 300;
        // for (int i = 0; i < b_hist.rows; ++i) {
        //     for (int j = 0; j < b_hist.cols; ++j) {
        //         printf("before  %f \n", b_hist.at<float>(i, j));
        //         b_hist.at<float>(i, j) = ((b_hist.at<float>(i, j)) * offset / (max));
        //         printf("after  %f \n", b_hist.at<float>(i, j));
        //     }
        // }
        //
        // int total = 0;
        // for (int j = 0; j < b_hist.rows; ++j) {
        //     for (int i = 0; i < b_hist.cols; ++i) {
        //         printf("1111  %d, %d \n", j, i);
        //         printf("222 ========== %f \n", b_hist.at<float>(j, i));
        //         total += b_hist.at<float>(j, i);
        //     }
        // }
        // printf("67777  %d \n", total);
        // total = 0;
        // for (int j = 0; j < g_hist.rows; ++j) {
        //     for (int i = 0; i < g_hist.cols; ++i) {
        //         printf("44444  %d, %d \n", j, i);
        //         printf("55555 ========== %f \n", g_hist.at<float>(j, i));
        //         total += g_hist.at<float>(j, i);
        //     }
        // }
        // printf("67777  %d \n", total);
    }

    // 2、绘制直方图
    // 定义直方图的参数
    int width = 512;//定义直方图的宽度
    int height = 300;//定义直方图的长度
    // 直方图白板
    Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));
    // 归一化直方图的高度
    // enum NormTypes   NORM_MINMAX    = 32 //!< flag
    // normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
    // 手写MIN-MAX归一化，效果与OpenCV normalize源码一致！
    my_normalize(b_hist, b_hist, 0, height, NORM_MINMAX);

    // 使用line绘制直方图
    int binStep = cvRound((float) width / (float) numbins);
    for (int i = 1; i < numbins; i++) {
        line(histImage,
             Point(binStep * (i - 1), height - cvRound(b_hist.at<float>(i - 1, 0))),
             Point(binStep * (i), height - cvRound(b_hist.at<float>(i, 0))),
             Scalar(255, 255, 255));
    }
    imshow("Histogram", histImage);
    waitKey(0);
}

/**
 * 手写my_calcHist，返回Mat(*histSize, 1, CV_64FC1, Scalar(0))类型的Hist数组
 * 暂时未考虑histSize≠256和Mask，后续改进。
 * @param images
 * @param nimages
 * @param channels
 * @param dims
 * @param histSize
 * @param ranges
 * @return
 */
Mat my_calcHist(Mat *images, int nimages, int channels, int dims, int *histSize, const float **ranges) {
    Mat hist = Mat(*histSize, 1, CV_64FC1, Scalar(0));
    int offset = 256 / *histSize;
    printf("%d, %d \n", images[channels].rows, images[channels].cols);
    for (int j = 0; j < images[channels].rows; ++j) {
        for (int i = 0; i < images[channels].cols; ++i) {
            // int index = bgr[0].at<uchar>(j, i) / offset;
            int index = images[channels].at<uchar>(j, i);
            hist.at<float>(index, 0)++;
        }
    }
    return hist;
}

// 归一化公式 src-min(src))*(b‘-a‘)/(max(src)-min(src))+ a‘
Mat my_normalize(Mat src, Mat dst, int alpha, int beta, NormTypes types) {
    if (types == NORM_MINMAX) {
        float max = -1, min = 9999999;
        float offset = beta - alpha;
        int channels_flag = 0;
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                if (channels_flag == 0) {
                    auto value = src.at<float>(i, j);
                    if (value > max) {
                        max = value;
                    }
                    if (value < min) {
                        min = value;
                    }
                } else {
                    // to do
                }
            };
        }
        printf("max-min  %f, %f \n", max, min);
        // 归一化公式 src-min(src))*(b‘-a‘)/(max(src)-min(src))+ a‘
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                dst.at<float>(i, j) = ((src.at<float>(i, j) - min) * offset / (max - min));
            }
        }
        return dst;
    }
}

void normalnization_equalization_hist_gray_image(string image_path) {
    normalnization_equalization_hist_color_image(image_path, 0);
}

void normalnization_equalization_hist_color_image(string image_path, int flag) {
    Mat srcImage = imread(image_path, flag);
    if (!srcImage.data) {
        printf("读取图片失败");
    }
    imshow("原图", srcImage);
    waitKey(0);
    // 转化图像为HSV
    Mat hsv;
    Mat result;

    if (flag == 1) {
        cvtColor(srcImage, hsv, COLOR_BGR2HSV);
    } else {
        hsv = srcImage;
    }
    vector<Mat> channels;
    split(hsv, channels);
    // 只需要获取图像的亮度Intensity，将其均衡化
    // equalizeHist(channels[0], channels[0]);
    // 手写equalizeHist，效果与源码相近！！
    myEqualizeHist(channels[0], channels[0]);

    // 合并结果通道，将结果转化为BGR
    merge(channels, hsv);
    if (flag == 1) {
        cvtColor(hsv, result, COLOR_HSV2BGR);
    } else {
        result = hsv;
    }
    imshow("Equalize", result);
    waitKey(0);
}

Mat myEqualizeHist(Mat src, Mat dst) {
    // 灰度分布直方图
    int hist[256] = {0,};
    // 色彩转换表，由src转化为dst的色彩对照表
    int lut[256];
    // 获取图像Hist信息
    for (int j = 0; j < src.rows; ++j) {
        for (int i = 0; i < src.cols; ++i) {
            int temp_index = src.ptr<uchar>(j)[i];
            // printf("temp_index = %d \n", temp_index);
            hist[temp_index]++;
        }
    }
    // for (int k = 0; k < 256; ++k) {
    //     printf("hist[i] = %d \n", hist[k]);
    // }
    // total等于像素点总数
    float total = (int) src.total();
    // 灰度分布密度
    // hist[i] / total
    float temp[256];
    for (int k = 0; k < 256; ++k) {
        // printf("hist[i] = %d \n", hist[k]);
        // temp[k]为累计分布函数，处理后像素值会均匀分布。
        // 累积分布函数是单调增函数（控制大小关系），并且值域是0到1（控制越界问题），
        // 所以直方图均衡化中使用的是累积分布函数。
        if (k == 0) {
            temp[k] = hist[k] / total;

        } else {
            temp[k] = temp[k - 1] + hist[k] / total;
        }
        lut[k] = (int) (255.0f * temp[k]);
    }

    for (int j = 0; j < dst.rows; ++j) {
        for (int i = 0; i < dst.cols; ++i) {
            int temp_index = src.ptr<uchar>(j)[i];
            dst.at<uchar>(j, i) = (uchar) lut[temp_index];
        }
    }
    return dst;
}


/**
 * 使用API的方法
 */
int MeanFilter_Gray(int a, int b) {
    Mat image, meanRes;
    image = imread(image_path + "lena.png", 0); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.

    blur(image, meanRes, Size(a, b));            //均值滤波

    namedWindow("均值滤波", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("均值滤波", meanRes);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}

int MeanFilter_Color(int a, int b) {
    Mat image, meanRes;
    image = imread(image_path + "lena.png", 1); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.

    blur(image, meanRes, Size(a, b));            //均值滤波

    namedWindow("均值滤波", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("均值滤波", meanRes);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}

int GaussianFilter_Gray(int a, int b) {
    Mat image, res;
    image = imread(image_path + "lena.png", 0); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.

    GaussianBlur(image, res, Size(a, b), 1);

    namedWindow("高斯滤波", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("高斯滤波", res);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}

int GaussianFilter_Color(int a, int b) {
    Mat image, res;
    image = imread(image_path + "lena.png", 1); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.

    GaussianBlur(image, res, Size(a, b), 1);

    namedWindow("高斯滤波", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("高斯滤波", res);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}


int Sobel() {
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat image, res;
    image = imread(image_path + "lena.png", 0); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.

    Sobel(image, grad_x, image.depth(), 1, 0, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    imshow("【效果图】 X方向Sobel", abs_grad_x);
    waitKey(0); // Wait for a keystroke in the window

    Sobel(image, grad_y, image.depth(), 0, 1, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    imshow("【效果图】Y方向Sobel", abs_grad_y);
    waitKey(0); // Wait for a keystroke in the window

    //【5】合并梯度(近似)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, res);
    imshow("【效果图】整体方向Sobel", res);
    waitKey(0); // Wait for a keystroke in the window

    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}

int Sobel_Color() {
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat image, res;
    image = imread(image_path + "lena.png", 1); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window

    Sobel(image, grad_x, image.depth(), 1, 0, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    imshow("【效果图】 X方向Sobel", abs_grad_x);
    waitKey(0); // Wait for a keystroke in the window

    Sobel(image, grad_y, image.depth(), 0, 1, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    imshow("【效果图】Y方向Sobel", abs_grad_y);
    waitKey(0); // Wait for a keystroke in the window

    //【5】合并梯度(近似)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, res);
    imshow("【效果图】整体方向Sobel", res);

    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}

//拉普拉斯模板
int Laplacian_Color() {
    Mat image, res;
    image = imread(image_path + "lena.png", IMREAD_COLOR); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window

    Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
    filter2D(image, res, image.depth(), kernel);

    namedWindow("拉普拉斯模板", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("拉普拉斯模板", res);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}


int Laplacian_Gray() {
    Mat image, res;
    image = imread(image_path + "lena.png", 0); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window

    Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
    filter2D(image, res, image.depth(), kernel);

    namedWindow("拉普拉斯模板", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("拉普拉斯模板", res);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}


int Lap2() {
    Mat image, res;
    image = imread(image_path + "lena.png", IMREAD_COLOR); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.

    res.create(image.size(), image.type());//为输出图像分配内容

    /*拉普拉斯滤波核3*3
     0  -1   0
    -1   5  -1
     0  -1   0  */
    //处理除最外围一圈外的所有像素值

    for (int i = 1; i < image.rows - 1; i++) {
        const uchar *pre = image.ptr<const uchar>(i - 1);//前一行
        const uchar *cur = image.ptr<const uchar>(i);//当前行，第i行
        const uchar *next = image.ptr<const uchar>(i + 1);//下一行
        uchar *output = res.ptr<uchar>(i);//输出图像的第i行
        int ch = image.channels();//通道个数
        int startCol = ch;//每一行的开始处理点
        int endCol = (image.cols - 1) * ch;//每一行的处理结束点
        for (int j = startCol; j < endCol; j++) {
            //输出图像的遍历指针与当前行的指针同步递增, 以每行的每一个像素点的每一个通道值为一个递增量, 因为要

            //考虑到图像的通道数
            //saturate_cast<uchar>保证结果在uchar范围内
            *output++ = saturate_cast<uchar>(5 * cur[j] - pre[j] - next[j] - cur[j - ch] - cur[j + ch]);
        }
    }
    //将最外围一圈的像素值设为0
    res.row(0).setTo(Scalar(0));
    res.row(res.rows - 1).setTo(Scalar(0));
    res.col(0).setTo(Scalar(0));
    res.col(res.cols - 1).setTo(Scalar(0));
    /*/或者也可以尝试将最外围一圈设置为原图的像素值
    image.row(0).copyTo(result.row(0));
    image.row(image.rows-1).copyTo(result.row(result.rows-1));
    image.col(0).copyTo(result.col(0));
    image.col(image.cols-1).copyTo(result.col(result.cols-1));*/
    namedWindow("拉普拉斯模板-手写", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("拉普拉斯模板-手写", res);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}

int Robert_G() {
    Mat image, res;
    image = imread(image_path + "lena.png", 0); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.
    res = image.clone();

    for (int i = 0; i < image.rows - 1; i++) {
        for (int j = 0; j < image.cols - 1; j++) {
            //根据公式计算
            int t1 = (image.at<uchar>(i, j) -
                      image.at<uchar>(i + 1, j + 1)) *
                     (image.at<uchar>(i, j) -
                      image.at<uchar>(i + 1, j + 1));
            int t2 = (image.at<uchar>(i + 1, j) -
                      image.at<uchar>(i, j + 1)) *
                     (image.at<uchar>(i + 1, j) -
                      image.at<uchar>(i, j + 1));
            //计算g（x,y）
            res.at<uchar>(i, j) = (uchar) sqrt(t1 + t2);
        }
    }

    namedWindow("Robert_G滤波", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Robert_G滤波", res);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;

}


void
EnhanceFilter(Mat img, Mat &dst, double dProportion, int nTempH, int nTempW, int nTempMY, int nTempMX, float *pfArray,
              float fCoef) {


    int i, j, nHeight = img.rows, nWidth = img.cols;
    vector<vector<int>> GrayMat1, GrayMat2, GrayMat3;//暂存按比例叠加图像，R,G,B三通道
    vector<int> vecRow1(nWidth, 0), vecRow2(nWidth, 0), vecRow3(nWidth, 0);
    for (i = 0; i < nHeight; i++) {
        GrayMat1.push_back(vecRow1);
        GrayMat2.push_back(vecRow2);
        GrayMat3.push_back(vecRow3);
    }

    //锐化图像，输出带符号响应，并与原图像按比例叠加
    for (i = nTempMY; i < nHeight - (nTempH - nTempMY) + 1; i++) {
        for (j = nTempMX; j < nWidth - (nTempW - nTempMX) + 1; j++) {
            float fResult1 = 0;
            float fResult2 = 0;
            float fResult3 = 0;
            for (int k = 0; k < nTempH; k++) {
                for (int l = 0; l < nTempW; l++) {
                    //分别计算三通道加权和
                    fResult1 += img.at<Vec3b>(i, j)[0] * pfArray[k * nTempW + 1];
                    fResult2 += img.at<Vec3b>(i, j)[1] * pfArray[k * nTempW + 1];
                    fResult3 += img.at<Vec3b>(i, j)[2] * pfArray[k * nTempW + 1];
                }
            }

            //三通道加权和分别乘以系数并限制响应范围，最后和原图像按比例混合
            fResult1 *= fCoef;
            if (fResult1 > 255)
                fResult1 = 255;
            if (fResult1 < -255)
                fResult1 = -255;
            GrayMat1[i][j] = dProportion * img.at<Vec3b>(i, j)[0] + fResult1 + 0.5;

            fResult2 *= fCoef;
            if (fResult2 > 255)
                fResult2 = 255;
            if (fResult2 < -255)
                fResult2 = -255;
            GrayMat2[i][j] = dProportion * img.at<Vec3b>(i, j)[1] + fResult2 + 0.5;

            fResult3 *= fCoef;
            if (fResult3 > 255)
                fResult3 = 255;
            if (fResult3 < -255)
                fResult3 = -255;
            GrayMat3[i][j] = dProportion * img.at<Vec3b>(i, j)[2] + fResult3 + 0.5;
        }
    }
    int nMax1 = 0, nMax2 = 0, nMax3 = 0;//三通道最大灰度和值
    int nMin1 = 65535, nMin2 = 65535, nMin3 = 65535;//三通道最小灰度和值
    //分别统计三通道最大值最小值
    for (i = nTempMY; i < nHeight - (nTempH - nTempMY) + 1; i++) {
        for (j = nTempMX; j < nWidth - (nTempW - nTempMX) + 1; j++) {
            if (GrayMat1[i][j] > nMax1)
                nMax1 = GrayMat1[i][j];
            if (GrayMat1[i][j] < nMin1)
                nMin1 = GrayMat1[i][j];

            if (GrayMat2[i][j] > nMax2)
                nMax2 = GrayMat2[i][j];
            if (GrayMat2[i][j] < nMin2)
                nMin2 = GrayMat2[i][j];

            if (GrayMat3[i][j] > nMax3)
                nMax3 = GrayMat3[i][j];
            if (GrayMat3[i][j] < nMin3)
                nMin3 = GrayMat3[i][j];
        }
    }
    //将按比例叠加后的三通道图像取值范围重新归一化到[0,255]
    int nSpan1 = nMax1 - nMin1, nSpan2 = nMax2 - nMin2, nSpan3 = nMax3 - nMin3;
    for (i = nTempMY; i < nHeight - (nTempH - nTempMY) + 1; i++) {
        for (j = nTempMX; j < nWidth - (nTempW - nTempMX) + 1; j++) {
            int br, bg, bb;
            if (nSpan1 > 0)
                br = (GrayMat1[i][j] - nMin1) * 255 / nSpan1;
            else if (GrayMat1[i][j] <= 255)
                br = GrayMat1[i][j];
            else
                br = 255;
            dst.at<Vec3b>(i, j)[0] = br;

            if (nSpan2 > 0)
                bg = (GrayMat2[i][j] - nMin2) * 255 / nSpan2;
            else if (GrayMat2[i][j] <= 255)
                bg = GrayMat2[i][j];
            else
                bg = 255;
            dst.at<Vec3b>(i, j)[1] = bg;

            if (nSpan3 > 0)
                bb = (GrayMat3[i][j] - nMin3) * 255 / nSpan3;
            else if (GrayMat3[i][j] <= 255)
                bb = GrayMat3[i][j];
            else
                bb = 255;
            dst.at<Vec3b>(i, j)[2] = bb;
        }
    }
}


/**
  * 1、利用均值模板平滑灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 3*3、5*5 和 9*9 尺寸的均值模板平滑灰度图像
  * 2、利用高斯模板平滑灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 3*3、5*5 和 9*9 尺寸的高斯模板平滑灰度图像
  * 3、利用 Laplacian、Robert、Sobel 模板锐化灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 Laplacian、Robert、 Sobel 模板锐化灰度图像
  * 4、利用高提升滤波算法增强灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，设计高提升滤波算法增 强图像
  * 5、利用均值模板平滑彩色图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，利 用 3*3、5*5 和 9*9 尺寸的均值模板平滑彩色图像
  * 6、利用高斯模板平滑彩色图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，分 别利用 3*3、5*5 和 9*9 尺寸的高斯模板平滑彩色图像
  * 7、利用 Laplacian、Robert、Sobel 模板锐化灰度图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，分 别利用 Laplacian、Robert、Sobel 模板锐化彩色图像
  *
  */
void lab3() {
    string image = image_path + "lena.png";

    // 1、利用均值模板平滑灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 3*3、5*5 和 9*9 尺寸的均值模板平滑灰度图像
    // 5、利用均值模板平滑彩色图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，利 用 3*3、5*5 和 9*9 尺寸的均值模板平滑彩色图像
    MeanFilter(image, 3, 3, 0);
    MeanFilter(image, 5, 5, 0);
    MeanFilter(image, 9, 9, 0);
    MeanFilter(image, 3, 3, 1);
    MeanFilter(image, 5, 5, 1);
    MeanFilter(image, 9, 9, 1);


    // 2、利用高斯模板平滑灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 3*3、5*5 和 9*9 尺寸的高斯模板平滑灰度图像
    // 6、利用高斯模板平滑彩色图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，分 别利用 3*3、5*5 和 9*9 尺寸的高斯模板平滑彩色图像
    double sigma = 0.849;
    // 对图像进行卷积，目前处理单通道的灰度图像，后续可以加for循环处理BGR的图像
    // CovImage中第二个参数dst是引用类型，直接对dst进行操作
    GaussianFilter(image, 3, 3, sigma, 0);
    GaussianFilter(image, 5, 5, sigma, 0);
    GaussianFilter(image, 9, 9, sigma, 0);
    GaussianFilter(image, 3, 3, sigma, 1);
    GaussianFilter(image, 5, 5, sigma, 1);
    GaussianFilter(image, 9, 9, sigma, 1);


    // 3、利用 Laplacian、Robert、Sobel 模板锐化灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 Laplacian、Robert、 Sobel 模板锐化灰度图像
    // 7、利用 Laplacian、Robert、Sobel 模板锐化灰度图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，分别利用 Laplacian、Robert、Sobel 模板锐化彩色图像
    // 目前的锐化模板都是使用提前定义好的通用模板，暂时只支持3*3的模板
    int m_width = 3;
    int m_height = 3;
    // 使用Robert模版对灰度图像和彩色图像进行锐化
    // 与调用API的方法进行比较，效果无明显差异
    RobertFilter(image, m_width, m_height, 0);
    RobertFilter(image, m_width, m_height, 1);
    // 与调用API的方法进行比较，效果无明显差异
    // 使用OpenCV的API进行效果比较
    // Robert_G();

    // 使用Sobel模版对灰度图像和彩色图像进行锐化
    SobelFilter(image, m_width, m_height, 0);
    SobelFilter(image, m_width, m_height, 1);
    // 使用OpenCV的API进行效果比较，效果无明显差异
    // Sobel_Color();

    // 使用Laplacian模版对灰度图像和彩色图像进行锐化
    LaplaceFilter(image, m_width, m_height, 0);
    LaplaceFilter(image, m_width, m_height, 1);
    // 使用OpenCV的API进行效果比较，效果无明显差异
    // Laplacian_Gray();
    // Laplacian_Color();


    // 4、利用高提升滤波算法增强灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，设计高提升滤波算法增强图像
    // 高提升滤波：
    // 一般锐化模板其系数之和均为0，这说明算子在灰度恒定区域的响应为0，
    // 即在锐化处理后的图像中，原图像的平滑区域近乎黑色，
    // 而原图中所有的边缘、细节和灰度跳变点都作为黑背景中的高灰度部分突出显示。
    // 基于锐化的图像增强中存储希望在增强边缘和细节的同时仍然保留原图像中的信息，而非将平滑区域的灰度信息丢失，
    // 因此可以把原图像加上锐化后的图像得到比较理想的结果。
    // 其原理流程图如下：
    // 1、图像锐化
    // 2、原图像与锐化图像按比例混合
    // 3、混合后的灰度调整（归一化至[0,255])
    LaplaceFilter(image, m_width, m_height, 0);
    EnhanceLaplaceFilter(image, m_width, m_height, 0);
    LaplaceFilter(image, m_width, m_height, 1);
    EnhanceLaplaceFilter(image, m_width, m_height, 1);

    SobelFilter(image, m_width, m_height, 0);
    EnhanceSobelFilter(image, m_width, m_height, 0);
    SobelFilter(image, m_width, m_height, 1);
    EnhanceSobelFilter(image, m_width, m_height, 1);

    RobertFilter(image, m_width, m_height, 0);
    EnhanceRobertFilter(image, m_width, m_height, 0);
    RobertFilter(image, m_width, m_height, 1);
    EnhanceRobertFilter(image, m_width, m_height, 1);

    // 除此之外可以通过修改拉普拉斯的模版，
    // 例如double laplace[] = {0, -1, 0, -1, 5, -1, 0, -1, 0}
    // 进行高频提升过滤
}


// 根据mask模板，对图像进行卷积，此方法不对图像进行Padding填充，因此会缩小图像
void CovImage(Mat src, Mat &dst, double *mask, int m_width, int m_height) {
    // 1、获取模板中心点偏移坐标
    int mask_center_h = m_height / 2;
    int mask_center_w = m_width / 2;
    int height = src.rows;
    int width = src.cols;
    // 2、对图像的像素点进行卷积
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double value = 0.0;
            // 2.1、防止数组越界问题，不对最外围的像素点进行处理
            if ((i - mask_center_h) >= 0 && (j - mask_center_w) >= 0 && (i + mask_center_h) < height &&
                (j + mask_center_w) < width) {
                // 2.2、使用mask模板对图像进行卷积
                for (int n = 0; n < m_height; n++) {
                    for (int m = 0; m < m_width; m++) {
                        value += src.at<uchar>(i + n - mask_center_h, j + m - mask_center_w) * mask[n * m_width + m];
                    }
                }
            }
            // 3、处理完对值域不在0～255的值进行处理
            if (value > 255)
                value = 255;
            if (value < 0)
                value = 0;
            dst.at<uchar>(i, j) = value;
        }
    }
}

// 高提升滤波：
// 一般锐化模板其系数之和均为0，这说明算子在灰度恒定区域的响应为0，
// 即在锐化处理后的图像中，原图像的平滑区域近乎黑色，
// 而原图中所有的边缘、细节和灰度跳变点都作为黑背景中的高灰度部分突出显示。
// 基于锐化的图像增强中存储希望在增强边缘和细节的同时仍然保留原图像中的信息，而非将平滑区域的灰度信息丢失，
// 因此可以把原图像加上锐化后的图像得到比较理想的结果。
// 其原理流程图如下：
// 1、图像锐化
// 2、原图像与锐化图像按比例混合
// 3、混合后的灰度调整（归一化至[0,255])
// 根据mask模板，对图像进行卷积，然后将原图像与锐化图像按比例混合，用于实现高提升滤波算法增强灰度图像
void EnhanceCovImage(Mat src, Mat &dst, double *mask, int m_width, int m_height) {
    Mat src2;
    src.copyTo(src2);
    int mask_center_h = m_height / 2;
    int mask_center_w = m_width / 2;
    int height = src.rows;
    int width = src.cols;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double value = 0.0;
            // 1、图像锐化，核心逻辑与CovImage一致，这里省略
            if ((i - mask_center_h) >= 0 && (j - mask_center_w) >= 0 && (i + mask_center_h) < height &&
                (j + mask_center_w) < width) {
                for (int n = 0; n < m_height; n++) {
                    for (int m = 0; m < m_width; m++) {
                        value += src.at<uchar>(i + n - mask_center_h, j + m - mask_center_w) * mask[n * m_width + m];
                    }
                }
            }
            if (value > 255)
                value = 255;
            if (value < 0)
                value = 0;
            // 2、原图像与锐化图像按比例混合
            double value2 = src.at<uchar>(i, j) + value;
            // 3、混合后的灰度调整（通过简单的二值域法处理)
            if (value2 > 255)
                value2 = 255;
            if (value2 < 0)
                value2 = 0;
            dst.at<uchar>(i, j) = value2;
        }
    }
}

// Laplace滤波器模板生成，目前固定生成3*3的固定Laplace算子
void LaplaceMask(double *mask, int width, int height) {
    // double laplace[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    double laplace[] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    // double laplace[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    // double laplace[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    // double laplace[] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
    for (int i = 0; i < width * height; i++) {
        mask[i] = laplace[i];
    }
    for (int i = 0; i < width * height; i++) {
        printf("mask[%d] = %f", i, mask[i]);
    }
}

// Robert滤波器模板生成，目前固定生成3*3的固定Robert算子
void RobertMask(double *mask, int width, int height) {
    double roberx[] = {0, 0, 0, 0, -1, 0, 0, 0, 1};
    // double robery[] = {0, 0, 0, 0, 0, -1, 0, 1, 0};
    for (int i = 0; i < width * height; i++) {
        mask[i] = roberx[i];
    }
    for (int i = 0; i < width * height; i++) {
        printf("mask[%d] = %f", i, mask[i]);
    }
}

// Robert滤波器模板生成，目前固定生成3*3的固定Robert算子
void RobertMaskY(double *mask, int width, int height) {
    // double roberx[] = {0, 0, 0, 0, -1, 0, 0, 0, 1};
    double robery[] = {0, 0, 0, 0, 0, -1, 0, 1, 0};
    for (int i = 0; i < width * height; i++) {
        mask[i] = robery[i];
    }
    for (int i = 0; i < width * height; i++) {
        printf("mask[%d] = %f", i, mask[i]);
    }
}

// Sobel滤波器模板生成，目前固定生成3*3的固定Sobel算子
void SobelMask(double *mask, int width, int height) {
    //          -1  -2  -1
    // mask x   0   0   0
    //          1   2   1
    //          -1  0   1
    // mask y   -2  0   2
    //          -1  0   1
    double sobelx[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};//sobelx
    // double sobely[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};//sobely
    for (int i = 0; i < width * height; i++) {
        mask[i] = sobelx[i];
    }
    for (int i = 0; i < width * height; i++) {
        printf("mask[%d] = %f", i, mask[i]);
    }
}

// Sobel滤波器模板生成，目前固定生成3*3的固定Robert算子
void SobelMaskY(double *mask, int width, int height) {
    //          -1  -2  -1
    // mask x   0   0   0
    //          1   2   1
    //          -1  0   1
    // mask y   -2  0   2
    //          -1  0   1
    // double sobelx[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};//sobelx
    double sobely[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};//sobely
    for (int i = 0; i < width * height; i++) {
        mask[i] = sobely[i];
    }
    for (int i = 0; i < width * height; i++) {
        printf("mask[%d] = %f", i, mask[i]);
    }
}

// 利用Laplace算子锐化灰度图像/彩色图像
Mat LaplaceFilter(string image_path, int m_width, int m_height, int flag) {
    Mat src = imread(image_path, flag);
    Mat dst;
    src.copyTo(dst);
    imshow("原图", src);
    waitKey(0);
    // 1、获取Laplace模板
    double *mask = new double[m_width * m_height];
    LaplaceMask(mask, m_width, m_height);
    // 2、判断灰度图还是彩色图
    if (flag == 1) {
        // 2.1、彩色图分别对BGR三个通道进行卷积，最后再合并
        vector<Mat> channels;
        split(src, channels);
        vector<Mat> channels_dst;
        split(dst, channels_dst);
        for (int i = 0; i < 3; ++i) {
            // 此处代码有异常，src和dst都传入同一个，导致对图像重复卷积
            // CovImage(channels[i], channels[i], mask, m_width, m_height);
            CovImage(channels[i], channels_dst[i], mask, m_width, m_height);
        }
        merge(channels_dst, dst);
    } else if (flag == 0) {
        // 2.2、对灰度图进行卷积
        CovImage(src, dst, mask, m_width, m_height);
    } else {
        // do nothing
    }
    imshow("滤波后", dst);
    waitKey(0);
    return dst;
}

Mat EnhanceLaplaceFilter(string image_path, int m_width, int m_height, int flag) {
    Mat src = imread(image_path, flag);
    Mat dst;
    src.copyTo(dst);
    imshow("原图", src);
    waitKey(0);

    double *mask = new double[m_width * m_height];
    LaplaceMask(mask, m_width, m_height);

    if (flag == 1) {
        vector<Mat> channels;
        split(src, channels);
        vector<Mat> channels_dst;
        split(dst, channels_dst);
        for (int i = 0; i < 3; ++i) {
            EnhanceCovImage(channels[i], channels_dst[i], mask, m_width, m_height);
        }
        merge(channels_dst, dst);
    } else if (flag == 0) {
        EnhanceCovImage(src, dst, mask, m_width, m_height);
    } else {
        // do nothing
    }
    imshow("滤波后", dst);
    waitKey(0);
    return dst;
}

Mat RobertFilter(string image_path, int m_width, int m_height, int flag) {
    Mat src = imread(image_path, flag);
    Mat dst;
    Mat dst_x;
    Mat dst_y;
    src.copyTo(dst);
    src.copyTo(dst_x);
    src.copyTo(dst_y);
    imshow("原图", src);
    waitKey(0);

    double *mask = new double[m_width * m_height];
    RobertMask(mask, m_width, m_height);
    double *mask_y = new double[m_width * m_height];
    RobertMaskY(mask_y, m_width, m_height);

    if (flag == 1) {
        vector<Mat> channels_x;
        split(dst_x, channels_x);
        vector<Mat> channels_y;
        split(dst_y, channels_y);
        vector<Mat> channels;
        split(dst, channels);

        int height = dst.rows;
        int width = dst.cols;

        for (int c = 0; c < 3; ++c) {
            CovImage(channels_x[c], channels_x[c], mask, m_width, m_height);
            CovImage(channels_y[c], channels_y[c], mask_y, m_width, m_height);
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    printf("channels_x[%d，%d] = %d \n", j, i, channels_x[c].at<uchar>(j, i));
                    printf("channels_y[%d，%d] = %d \n", j, i, channels_y[c].at<uchar>(j, i));
                    double value1 = channels_x[c].at<uchar>(j, i);
                    if (value1 > 255)
                        value1 = 255;
                    if (value1 < 0)
                        value1 = 0;
                    double value2 = channels_y[c].at<uchar>(j, i);
                    if (value2 > 255)
                        value2 = 255;
                    if (value2 < 0)
                        value2 = 0;
                    double value = value1 + value2;
                    if (value > 255)
                        value = 255;
                    if (value < 0)
                        value = 0;
                    channels[c].at<uchar>(j, i) = value;
                }
            }
        }
        merge(channels_x, dst_x);
        merge(channels_y, dst_y);
        merge(channels, dst);
    } else if (flag == 0) {
        CovImage(src, dst_x, mask, m_width, m_height);
        CovImage(src, dst_y, mask_y, m_width, m_height);
        int height = dst.rows;
        int width = dst.cols;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                // double value = (dst_x.at<uchar>(j, i) + dst_y.at<uchar>(j, i));
                double value = abs(dst_x.at<uchar>(j, i)) + abs(dst_y.at<uchar>(j, i));
                if (value > 255)
                    value = 255;
                if (value < 0)
                    value = 0;
                dst.at<uchar>(j, i) = value;
            }
        }
    } else {
        // do nothing
    }
    imshow("滤波后", dst_x);
    waitKey(0);
    imshow("滤波后", dst_y);
    waitKey(0);
    imshow("滤波后", dst);
    waitKey(0);
    return dst;
}

Mat EnhanceRobertFilter(string image_path, int m_width, int m_height, int flag) {
    Mat src = imread(image_path, flag);
    Mat dst;
    Mat dst_x;
    Mat dst_y;
    src.copyTo(dst);
    src.copyTo(dst_x);
    src.copyTo(dst_y);
    imshow("原图", src);
    waitKey(0);

    double *mask = new double[m_width * m_height];
    RobertMask(mask, m_width, m_height);
    double *mask_y = new double[m_width * m_height];
    RobertMaskY(mask_y, m_width, m_height);

    if (flag == 1) {
        vector<Mat> channels_x;
        split(dst_x, channels_x);
        vector<Mat> channels_y;
        split(dst_y, channels_y);
        vector<Mat> channels;
        split(dst, channels);

        int height = dst.rows;
        int width = dst.cols;

        for (int c = 0; c < 3; ++c) {
            EnhanceCovImage(channels_x[c], channels_x[c], mask, m_width, m_height);
            EnhanceCovImage(channels_y[c], channels_y[c], mask_y, m_width, m_height);
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    printf("channels_x[%d，%d] = %d \n", j, i, channels_x[c].at<uchar>(j, i));
                    printf("channels_y[%d，%d] = %d \n", j, i, channels_y[c].at<uchar>(j, i));
                    double value1 = channels_x[c].at<uchar>(j, i);
                    if (value1 > 255)
                        value1 = 255;
                    if (value1 < 0)
                        value1 = 0;
                    double value2 = channels_y[c].at<uchar>(j, i);
                    if (value2 > 255)
                        value2 = 255;
                    if (value2 < 0)
                        value2 = 0;
                    double value = value1 + value2;
                    if (value > 255)
                        value = 255;
                    if (value < 0)
                        value = 0;
                    channels[c].at<uchar>(j, i) = value;
                }
            }
        }
        merge(channels_x, dst_x);
        merge(channels_y, dst_y);
        merge(channels, dst);
    } else if (flag == 0) {
        EnhanceCovImage(src, dst_x, mask, m_width, m_height);
        EnhanceCovImage(src, dst_y, mask_y, m_width, m_height);
        int height = dst.rows;
        int width = dst.cols;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                // double value = (dst_x.at<uchar>(j, i) + dst_y.at<uchar>(j, i));
                double value = abs(dst_x.at<uchar>(j, i)) + abs(dst_y.at<uchar>(j, i));
                if (value > 255)
                    value = 255;
                if (value < 0)
                    value = 0;
                dst.at<uchar>(j, i) = value;
            }
        }
    } else {
        // do nothing
    }
    imshow("滤波后", dst_x);
    waitKey(0);
    imshow("滤波后", dst_y);
    waitKey(0);
    imshow("滤波后", dst);
    waitKey(0);
    return dst;
}


Mat SobelFilter(string image_path, int m_width, int m_height, int flag) {
    Mat src = imread(image_path, flag);
    Mat dst;
    Mat dst_x;
    Mat dst_y;
    src.copyTo(dst);
    src.copyTo(dst_x);
    src.copyTo(dst_y);
    imshow("原图", src);
    waitKey(0);

    double *mask = new double[m_width * m_height];
    SobelMask(mask, m_width, m_height);
    double *mask_y = new double[m_width * m_height];
    SobelMaskY(mask_y, m_width, m_height);

    if (flag == 1) {
        vector<Mat> channels_x;
        split(dst_x, channels_x);
        vector<Mat> channels_y;
        split(dst_y, channels_y);
        vector<Mat> channels_x_dst;
        split(dst_x, channels_x_dst);
        vector<Mat> channels_y_dst;
        split(dst_y, channels_y_dst);
        vector<Mat> channels;
        split(dst, channels);

        int height = dst.rows;
        int width = dst.cols;

        for (int c = 0; c < 3; ++c) {
            CovImage(channels_x[c], channels_x_dst[c], mask, m_width, m_height);
            CovImage(channels_y[c], channels_y_dst[c], mask_y, m_width, m_height);
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    printf("channels_x[%d，%d] = %d \n", j, i, channels_x_dst[c].at<uchar>(j, i));
                    printf("channels_y[%d，%d] = %d \n", j, i, channels_y_dst[c].at<uchar>(j, i));
                    double value1 = channels_x_dst[c].at<uchar>(j, i);
                    if (value1 > 255)
                        value1 = 255;
                    if (value1 < 0)
                        value1 = 0;
                    double value2 = channels_y_dst[c].at<uchar>(j, i);
                    if (value2 > 255)
                        value2 = 255;
                    if (value2 < 0)
                        value2 = 0;
                    double value = value1 + value2;
                    if (value > 255)
                        value = 255;
                    if (value < 0)
                        value = 0;
                    channels[c].at<uchar>(j, i) = value;
                }
            }
        }
        merge(channels_x_dst, dst_x);
        merge(channels_y_dst, dst_y);
        merge(channels, dst);
    } else if (flag == 0) {
        CovImage(src, dst_x, mask, m_width, m_height);
        CovImage(src, dst_y, mask_y, m_width, m_height);
        int height = dst.rows;
        int width = dst.cols;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                // double value = (dst_x.at<uchar>(j, i) + dst_y.at<uchar>(j, i));
                double value = abs(dst_x.at<uchar>(j, i)) + abs(dst_y.at<uchar>(j, i));
                if (value > 255)
                    value = 255;
                if (value < 0)
                    value = 0;
                dst.at<uchar>(j, i) = value;
            }
        }
    } else {
        // do nothing
    }
    imshow("滤波后", dst_x);
    waitKey(0);
    imshow("滤波后", dst_y);
    waitKey(0);
    imshow("滤波后", dst);
    waitKey(0);
    return dst;
}

Mat EnhanceSobelFilter(string image_path, int m_width, int m_height, int flag) {
    Mat src = imread(image_path, flag);
    Mat dst;
    Mat dst_x;
    Mat dst_y;
    src.copyTo(dst);
    src.copyTo(dst_x);
    src.copyTo(dst_y);
    imshow("原图", src);
    waitKey(0);

    double *mask = new double[m_width * m_height];
    SobelMask(mask, m_width, m_height);
    double *mask_y = new double[m_width * m_height];
    SobelMaskY(mask_y, m_width, m_height);

    if (flag == 1) {
        vector<Mat> channels_x;
        split(dst_x, channels_x);
        vector<Mat> channels_y;
        split(dst_y, channels_y);
        vector<Mat> channels_x_dst;
        split(dst_x, channels_x_dst);
        vector<Mat> channels_y_dst;
        split(dst_y, channels_y_dst);
        vector<Mat> channels;
        split(dst, channels);

        int height = dst.rows;
        int width = dst.cols;

        for (int c = 0; c < 3; ++c) {
            EnhanceCovImage(channels_x[c], channels_x_dst[c], mask, m_width, m_height);
            EnhanceCovImage(channels_y[c], channels_y_dst[c], mask_y, m_width, m_height);
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    printf("channels_x[%d，%d] = %d \n", j, i, channels_x_dst[c].at<uchar>(j, i));
                    printf("channels_y[%d，%d] = %d \n", j, i, channels_y_dst[c].at<uchar>(j, i));
                    double value1 = channels_x_dst[c].at<uchar>(j, i);
                    if (value1 > 255)
                        value1 = 255;
                    if (value1 < 0)
                        value1 = 0;
                    double value2 = channels_y_dst[c].at<uchar>(j, i);
                    if (value2 > 255)
                        value2 = 255;
                    if (value2 < 0)
                        value2 = 0;
                    double value = value1 + value2;
                    if (value > 255)
                        value = 255;
                    if (value < 0)
                        value = 0;
                    channels[c].at<uchar>(j, i) = value;
                }
            }
        }
        merge(channels_x_dst, dst_x);
        merge(channels_y_dst, dst_y);
        merge(channels, dst);
    } else if (flag == 0) {
        EnhanceCovImage(src, dst_x, mask, m_width, m_height);
        EnhanceCovImage(src, dst_y, mask_y, m_width, m_height);
        int height = dst.rows;
        int width = dst.cols;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                // double value = (dst_x.at<uchar>(j, i) + dst_y.at<uchar>(j, i));
                double value = abs(dst_x.at<uchar>(j, i)) + abs(dst_y.at<uchar>(j, i));
                if (value > 255)
                    value = 255;
                if (value < 0)
                    value = 0;
                dst.at<uchar>(j, i) = value;
            }
        }
    } else {
        // do nothing
    }
    imshow("滤波后", dst_x);
    waitKey(0);
    imshow("滤波后", dst_y);
    waitKey(0);
    imshow("滤波后", dst);
    waitKey(0);
    return dst;
}


// 生成width * height的均值模板，均值模板公式：1.0 / (width * height);
void MeanMask(double *mask, int width, int height) {
    double meanvalue = 1.0 / (width * height);
    for (int i = 0; i < width * height; i++) {
        mask[i] = meanvalue;
    }
}

// 利用均值模板平滑灰度图像/彩色图像
Mat MeanFilter(string image_path, int m_width, int m_height, int flag) {
    Mat src = imread(image_path, flag);
    Mat dst;
    src.copyTo(dst);
    imshow("原图", src);
    waitKey(0);

    double *mask = new double[m_width * m_height];
    MeanMask(mask, m_width, m_height);
    if (flag == 1) {
        vector<Mat> channels;
        split(src, channels);
        for (int i = 0; i < 3; ++i) {
            CovImage(channels[i], channels[i], mask, m_width, m_height);
        }
        merge(channels, dst);
    }
    if (flag == 0) {
        CovImage(src, dst, mask, m_width, m_height);
    }
    imshow("滤波后", dst);
    waitKey(0);
    return dst;
}

// 利用高斯模板平滑灰度图像/彩色图像
Mat GaussianFilter(string image_path, int m_width, int m_height, double sigma, int flag) {
    Mat src = imread(image_path, flag);
    Mat dst;
    src.copyTo(dst);
    imshow("原图", src);
    waitKey(0);
    // 1、生成高斯模板
    double *mask = new double[m_width * m_height];
    GaussianMask(mask, m_width, m_height, sigma);
    // 2、判断灰度图还是彩色图
    if (flag == 1) {
        // 2.1、彩色图分别对BGR三个通道进行卷积，最后再合并
        vector<Mat> channels;
        split(src, channels);
        for (int i = 0; i < 3; ++i) {
            CovImage(channels[i], channels[i], mask, m_width, m_height);
        }
        merge(channels, dst);
    }
    if (flag == 0) {
        // 2.2、对灰度图进行卷积
        CovImage(src, dst, mask, m_width, m_height);
    }
    imshow("滤波后", dst);
    waitKey(0);
    return dst;
}

/**
 * 生成width * height的高斯模板，高斯模板公式：G(x,y) = e ^ (-(x^2+y^2) / 2 * sigma^2)
 * 高斯模板公式：G(x,y) =
 *                  =   e ^ (-(x^2+y^2) / 2 * sigma^2)
 *                  =   e ^ (-(r^2) / 2 * sigma^2)
 * 默认定义sigma=0.849，
 * 参考 数字图像处理课件《灰度变换与空间滤波》 高斯半径为sigma=0.849
 * 3*3 归一化模版为
 *          1   2   1
 * 1/16 *   2   4   2
 *          1   2   1
 * @param mask
 * @param width
 * @param height
 * @param sigma
 */
void GaussianMask(double *mask, int width, int height, double sigma) {
    double sigmaPowTwo = sigma * sigma;
    double mask_center_h = (double) height / 2 - 0.5;
    double mask_center_w = (double) width / 2 - 0.5;
    double param = 1.0 / (2 * M_PI * sigmaPowTwo);
    double sum = 0.0;

    // 1、根据高斯模板公式获取初始模板
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double distance = sqrt(
                    (j - mask_center_w) * (j - mask_center_w) + (i - mask_center_h) * (i - mask_center_h));
            mask[i * width + j] = param * exp(-(distance * distance) / (2 * sigmaPowTwo));
            sum += mask[i * width + j];
        }
    }
    // 2、归一化
    for (int i = 0; i < height * width; i++) {
        mask[i] /= sum;
    }
    for (int i = 0; i < height * width; i++) {
        printf("mask[%d] = %f \n", i, mask[i]);
    }
}

//中值滤波器
void MedianFilter(double *src, double *dst, int width, int height, int m_width, int m_height) {
    int mask_center_h = m_height / 2;
    int mask_center_w = m_width / 2;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double value = 0.0;
            //确保不会越界
            if ((i - mask_center_h) >= 0 && (j - mask_center_w) >= 0 && (i + mask_center_h) < height &&
                (j + mask_center_w) < width) {
                //对从src[(i-mask_center_h)*width+(j-mask_center_w)]开始的m_height*m_width个元素进行排序取中值
                int index = 0;
                double *sort = new double[m_width * m_height];
                for (int n = 0; n < m_height; n++) {
                    for (int m = 0; m < m_width; m++) {
                        sort[index] = src[(i + n - mask_center_h) * width + (j + m - mask_center_w)];
                        index++;
                    }
                }
                for (int n = 0; n < index - 1; n++) {
                    for (int m = 0; m < index - 1 - n; m++) {
                        if (sort[m] > sort[m + 1]) {
                            double temp_sort = sort[m];
                            sort[m] = sort[m + 1];
                            sort[m + 1] = temp_sort;
                        }
                    }
                }
                value = sort[index / 2];
            }

            dst[i * width + j] = value;
        }
    }
}


//本函数加入盐噪声
void addSalt(Mat &image, int n) {
    srand((unsigned) time(NULL));
    int i, j;
    for (int k = 0; k < n; k++)//将图像中n个像素随机置零
    {
        i = rand() % image.cols;
        j = rand() % image.rows;
        //将图像颜色随机改变
        if (image.channels() == 1)
            image.at<uchar>(j, i) = 255;
        else {
            for (int t = 0; t < image.channels(); t++) {
                image.at<Vec3b>(j, i)[t] = 255;
            }
        }
    }
}

void addPepper(Mat &image, int n)//本函数加入椒噪声
{
    srand((unsigned) time(NULL));
    for (int k = 0; k < n; k++)//将图像中n个像素随机置零
    {
        int i = rand() % image.cols;
        int j = rand() % image.rows;
        //将图像颜色随机改变
        if (image.channels() == 1)
            image.at<uchar>(j, i) = 0;
        else {
            for (int t = 0; t < image.channels(); t++) {
                image.at<Vec3b>(j, i)[t] = 0;
            }
        }

    }
}

int GaussianNoise(double mu, double sigma) {
    //定义一个特别小的值
    const double epsilon = numeric_limits<double>::min();//返回目标数据类型能表示的最逼近1的正数和1的差的绝对值
    static double z0, z1;
    static bool flag = false;
    flag = !flag;
    //flag为假，构造高斯随机变量
    if (!flag)
        return z1 * sigma + mu;
    double u1, u2;
    //构造随机变量

    do {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    //flag为真构造高斯随机变量X
    z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
    return z1 * sigma + mu;
}

Mat addGaussianNoise(Mat &srcImage) {
    Mat resultImage = srcImage.clone();    //深拷贝,克隆
    int channels = resultImage.channels();    //获取图像的通道
    int nRows = resultImage.rows;    //图像的行数

    int nCols = resultImage.cols * channels;   //图像的总列数
    //判断图像的连续性
    if (resultImage.isContinuous())    //判断矩阵是否连续，若连续，我们相当于只需要遍历一个一维数组
    {
        nCols *= nRows;
        nRows = 1;
    }
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {    //添加高斯噪声
            int val = resultImage.ptr<uchar>(i)[j] + GaussianNoise(2, 0.8) * 32;
            if (val < 0)
                val = 0;
            if (val > 255)
                val = 255;
            resultImage.ptr<uchar>(i)[j] = (uchar) val;
        }
    }
    return resultImage;
}

//中值滤波器
void medeanFilter(Mat &src, int win_size) {
    int rows = src.rows, cols = src.cols;
    int start = win_size / 2;
    for (int m = start; m < rows - start; m++) {
        for (int n = start; n < cols - start; n++) {
            vector<uchar> model;
            for (int i = -start + m; i <= start + m; i++) {
                for (int j = -start + n; j <= start + n; j++) {
                    //cout << int(src.at<uchar>(i, j)) << endl;
                    model.push_back(src.at<uchar>(i, j));
                }
            }
            sort(model.begin(), model.end());     //采用快速排序进行
            src.at<uchar>(m, n) = model[win_size * win_size / 2];
        }
    }
}

//算术均值滤波器
void meanFilter(Mat &src, int win_size) {
    int rows = src.rows, cols = src.cols;
    int start = win_size / 2;
    for (int m = start; m < rows - start; m++) {
        for (int n = start; n < cols - start; n++) {
            if (src.channels() == 1)                //灰色图
            {
                int sum = 0;
                for (int i = -start + m; i <= start + m; i++) {
                    for (int j = -start + n; j <= start + n; j++) {
                        sum += src.at<uchar>(i, j);
                    }
                }
                src.at<uchar>(m, n) = uchar(sum / win_size / win_size);
            } else {
                Vec3b pixel;
                int sum1[3] = {0};
                for (int i = -start + m; i <= start + m; i++) {
                    for (int j = -start + n; j <= start + n; j++) {
                        pixel = src.at<Vec3b>(i, j);
                        for (int k = 0; k < src.channels(); k++) {
                            sum1[k] += pixel[k];
                        }
                    }

                }
                for (int k = 0; k < src.channels(); k++) {
                    pixel[k] = sum1[k] / win_size / win_size;
                }
                src.at<Vec3b>(m, n) = pixel;
            }
        }
    }
}

//几何均值滤波器
Mat GeometryMeanFilter(Mat src) {
    Mat dst = src.clone();
    int row, col;
    int h = src.rows;
    int w = src.cols;
    double mul;
    double dc;
    int mn;
    //计算每个像素的去噪后 color 值
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            if (src.channels() == 1)                //灰色图
            {
                mul = 1.0;
                mn = 0;
                //统计邻域内的几何平均值，邻域大小 5*5
                for (int m = -2; m <= 2; m++) {
                    row = i + m;
                    for (int n = -2; n <= 2; n++) {
                        col = j + n;
                        if (row >= 0 && row < h && col >= 0 && col < w) {
                            int s = src.at<uchar>(row, col);
                            mul = mul * (s == 0 ? 1 : s); //邻域内的非零像素点相乘，最小值设定为1
                            mn++;
                        }
                    }
                }
                //计算 1/mn 次方
                dc = pow(mul, 1.0 / mn);
                //统计成功赋给去噪后图像。
                int res = (int) dc;
                dst.at<uchar>(i, j) = res;
            } else {
                double multi[3] = {1.0, 1.0, 1.0};
                mn = 0;
                Vec3b pixel;

                for (int m = -2; m <= 2; m++) {
                    row = i + m;
                    for (int n = -2; n <= 2; n++) {
                        col = j + n;
                        if (row >= 0 && row < h && col >= 0 && col < w) {
                            pixel = src.at<Vec3b>(row, col);
                            for (int k = 0; k < src.channels(); k++) {
                                multi[k] = multi[k] * (pixel[k] == 0 ? 1 : pixel[k]);//邻域内的非零像素点相乘，最小值设定为1
                            }
                            mn++;
                        }
                    }
                }
                double d;
                for (int k = 0; k < src.channels(); k++) {
                    d = pow(multi[k], 1.0 / mn);
                    pixel[k] = (int) d;
                }
                dst.at<Vec3b>(i, j) = pixel;
            }
        }
    }
    return dst;
}

//谐波均值滤波器——模板大小 5*5
Mat HarmonicMeanFilter(Mat src) {
    //IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
    Mat dst = src.clone();
    int row, col;
    int h = src.rows;
    int w = src.cols;
    double sum;
    double dc;
    int mn;
    //计算每个像素的去噪后 color 值
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            sum = 0.0;
            mn = 0;
            //统计邻域,5*5 模板
            for (int m = -2; m <= 2; m++) {
                row = i + m;
                for (int n = -2; n <= 2; n++) {
                    col = j + n;
                    if (row >= 0 && row < h && col >= 0 && col < w) {
                        int s = src.at<uchar>(row, col);
                        sum = sum + (s == 0 ? 255 : 255.0 / s);                    //如果是0，设定为255
                        mn++;
                    }
                }
            }
            int d;
            dc = mn * 255.0 / sum;
            d = dc;
            //统计成功赋给去噪后图像。
            dst.at<uchar>(i, j) = d;
        }
    }
    return dst;
}

//逆谐波均值大小滤波器——模板大小 5*5
Mat InverseHarmonicMeanFilter(Mat src, double Q) {
    Mat dst = src.clone();
    int row, col;
    int h = src.rows;
    int w = src.cols;
    double sum;
    double sum1;
    double dc;
    //double Q = 2;
    //计算每个像素的去噪后 color 值
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            sum = 0.0;
            sum1 = 0.0;
            //统计邻域
            for (int m = -2; m <= 2; m++) {
                row = i + m;
                for (int n = -2; n <= 2; n++) {
                    col = j + n;
                    if (row >= 0 && row < h && col >= 0 && col < w) {

                        int s = src.at<uchar>(row, col);
                        sum = sum + pow(s, Q + 1);
                        sum1 = sum1 + pow(s, Q);
                    }
                }
            }
            //计算 1/mn 次方
            int d;
            dc = sum1 == 0 ? 0 : (sum / sum1);
            d = (int) dc;
            //统计成功赋给去噪后图像。
            dst.at<uchar>(i, j) = d;
        }
    }
    return dst;
}

//自适应中值滤波
Mat SelfAdaptMedianFilter(Mat src) {
    Mat dst = src.clone();
    int row, col;
    int h = src.rows;
    int w = src.cols;
    double Zmin, Zmax, Zmed, Zxy, Smax = 7;
    int wsize;
    //计算每个像素的去噪后 color 值
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            //统计邻域
            wsize = 1;
            while (wsize <= 3) {
                Zmin = 255.0;
                Zmax = 0.0;
                Zmed = 0.0;
                int Zxy = src.at<uchar>(i, j);
                int mn = 0;
                for (int m = -wsize; m <= wsize; m++) {
                    row = i + m;
                    for (int n = -wsize; n <= wsize; n++) {
                        col = j + n;
                        if (row >= 0 && row < h && col >= 0 && col < w) {
                            int s = src.at<uchar>(row, col);
                            if (s > Zmax) {
                                Zmax = s;
                            }
                            if (s < Zmin) {
                                Zmin = s;
                            }
                            Zmed = Zmed + s;
                            mn++;
                        }
                    }
                }
                Zmed = Zmed / mn;
                int d;
                if ((Zmed - Zmin) > 0 && (Zmed - Zmax) < 0) {
                    if ((Zxy - Zmin) > 0 && (Zxy - Zmax) < 0) {
                        d = Zxy;
                    } else {
                        d = Zmed;
                    }
                    dst.at<uchar>(i, j) = d;
                    break;
                } else {
                    wsize++;
                    if (wsize > 3) {
                        int d;
                        d = Zmed;
                        dst.at<uchar>(i, j) = d;
                        break;
                    }
                }
            }
        }
    }
    return dst;
}

//自适应均值滤波
Mat SelfAdaptMeanFilter(Mat src) {
    Mat dst = src.clone();
    blur(src, dst, Size(7, 7));
    int row, col;
    int h = src.rows;
    int w = src.cols;
    int mn;
    double Zxy;
    double Zmed;
    double Sxy;
    double Sl;
    double Sn = 100;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int Zxy = src.at<uchar>(i, j);
            int Zmed = src.at<uchar>(i, j);
            Sl = 0;
            mn = 0;
            for (int m = -3; m <= 3; m++) {
                row = i + m;
                for (int n = -3; n <= 3; n++) {
                    col = j + n;
                    if (row >= 0 && row < h && col >= 0 && col < w) {
                        int Sxy = src.at<uchar>(row, col);
                        Sl = Sl + pow(Sxy - Zmed, 2);
                        mn++;
                    }
                }
            }
            Sl = Sl / mn;
            int d = (int) (Zxy - Sn / Sl * (Zxy - Zmed));
            dst.at<uchar>(i, j) = d;
        }
    }
    return dst;
}

IplImage *MatToIplImage(Mat image) {
    Mat t = image.clone();
    IplImage *res = &IplImage(t);
    return res;
}

Mat IplImageToMat(IplImage *image) {
    Mat res = cvarrToMat(image, true);
    return res;
}

void test1() {
    Mat image, noise, res;

    /*----------高斯噪声+算术均值-----------*/
    image = imread(image_path + "lena.png", 0); // Read the file
    imshow("原始图像", image);                // Show our image inside it.

    noise = addGaussianNoise(image);            //添加高斯噪声
    imshow("添加高斯噪声", noise);

    res = noise.clone();
    meanFilter(res, 5);                    //算术均值滤波器
    imshow("算术均值滤波器", res);
    /*------展示图像-------*/
    waitKey(0);
    destroyAllWindows();

    /*----------胡椒噪声+几何均值-----------*/
    image = imread(image_path + "lena.png", 0); // Read the file
    imshow("原始图像", image);                // Show our image inside it.

    noise = image.clone();
    addPepper(noise, 1000);
    imshow("添加1000个胡椒噪声", noise);

    res = noise.clone();
    meanFilter(res, 5);
    imshow("几何均值滤波器", res);

    /*------展示图像-------*/
    waitKey(0);
    destroyAllWindows();

    /*--------------盐噪声+谐波均值滤波器------------*/
    image = imread(image_path + "lena.png", 0); // Read the file
    imshow("原始图像", image);                // Show our image inside it.

    noise = image.clone();
    addSalt(noise, 1000);
    imshow("添加1000个盐噪声", noise);

    res = HarmonicMeanFilter(noise);
    imshow("5*5谐波均值滤波器", res);

    /*------展示图像-------*/
    waitKey(0);
    destroyAllWindows();


    /*-----------椒盐噪声+逆谐波均值滤波器-----------*/
    image = imread(image_path + "lena.png", 0); // Read the file
    imshow("原始图像", image);                // Show our image inside it.

    noise = image.clone();
    addSalt(noise, 1000);
    Sleep(2000);                    //防止随机数种子一样
    addPepper(noise, 1000);
    imshow("添加1000个盐噪声+1000个胡椒噪声", noise);

    res = InverseHarmonicMeanFilter(noise, 1);                //第二个参数是Q，Q=0退化成算术均值
    imshow("5*5逆谐波均值滤波器", res);

    /*------展示图像-------*/
    waitKey(0);
    destroyAllWindows();
    return;
}

void test2() {
    Mat image, noise, res1, res2;

    /*---------胡椒------------*/
    image = imread(image_path + "lena.png", 0); // Read the file
    imshow("原始图像", image);                // Show our image inside it.

    noise = image.clone();
    addPepper(noise, 1000);
    imshow("添加1000个胡椒噪声", noise);

    res1 = noise.clone();
    medeanFilter(res1, 5);
    imshow("5*5中均值滤波器", res1);

    res2 = noise.clone();

    res2 = noise.clone();
    medeanFilter(res2, 9);
    imshow("9*9中均值滤波器", res2);
    /*------展示图像-------*/
    waitKey(0);
    destroyAllWindows();

    /*-----------盐噪声---------------*/
    image = imread(image_path + "lena.png", 0); // Read the file
    imshow("原始图像", image);                // Show our image inside it.

    noise = image.clone();
    addSalt(noise, 1000);
    imshow("添加1000个盐噪声", noise);

    res1 = noise.clone();
    medeanFilter(res1, 5);
    imshow("5*5中均值滤波器", res1);

    res2 = noise.clone();

    res2 = noise.clone();
    medeanFilter(res2, 9);
    imshow("9*9中均值滤波器", res2);
    /*------展示图像-------*/
    waitKey(0);
    destroyAllWindows();


    /*-----------盐噪声+胡椒噪声---------------*/
    image = imread(image_path + "lena.png", 0); // Read the file
    imshow("原始图像", image);                // Show our image inside it.

    noise = image.clone();
    addSalt(noise, 1000);
    addPepper(noise, 1000);
    imshow("添加1000个盐噪声+1000个胡椒噪声", noise);

    res1 = noise.clone();
    medeanFilter(res1, 5);
    imshow("5*5中均值滤波器", res1);

    res2 = noise.clone();

    res2 = noise.clone();
    medeanFilter(res2, 9);
    imshow("9*9中均值滤波器", res2);
    /*------展示图像-------*/
    waitKey(0);
    destroyAllWindows();
    return;
}

void test3() {
    Mat image, res1, res2, noise;
    image = imread(image_path + "lena.png", 0); // Read the file
    imshow("原始图像", image);

    noise = image.clone();
    addPepper(noise, 1000);
    Sleep(2000);
    addSalt(noise, 1000);
    imshow("添加1000个胡椒噪声+1000个盐噪声", noise);

    res1 = SelfAdaptMeanFilter(image);
    imshow("自适应均值滤波", res1);

    res2 = noise.clone();
    meanFilter(res2, 7);
    imshow("7*7算术均值滤波", res2);

    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
}

void test4() {
    Mat image, res1, res2, noise;
    image = imread(image_path + "lena.png", 0); // Read the file
    imshow("原始图像", image);

    noise = image.clone();
    addPepper(noise, 1000);
    Sleep(2000);
    addSalt(noise, 1000);
    imshow("添加1000个胡椒噪声+1000个盐噪声", noise);

    res1 = SelfAdaptMedianFilter(image);
    imshow("自适应中值滤波", res1);

    res2 = noise.clone();
    medeanFilter(res2, 7);
    imshow("7*7中值滤波", res2);

    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
}

void test5() {
    Mat image, res1, res2, noise;
    image = imread(image_path + "lena.png", 1); // Read the file
    imshow("原始图像", image);

    noise = addGaussianNoise(image);
    imshow("添加高斯噪声", noise);

    res1 = noise.clone();
    meanFilter(res1, 5);                    //算术均值滤波器
    imshow("算术均值滤波器", res1);

    res2 = GeometryMeanFilter(noise);
    imshow("几何均值滤波器", res2);

    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();

}


void lab4() {
    test1();
    test2();
    test3();
    test4();
    test5();
    destroyAllWindows();
}