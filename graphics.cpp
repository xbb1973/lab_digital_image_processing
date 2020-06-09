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


// lab3负责调用实验2的内容逻辑
void lab3();


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

    lab3();

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
        if (k==0) {
            temp[k] = hist[k] / total;

        } else {
            temp[k] = temp[k-1] + hist[k] / total;
        }
        lut[k] = (int) ( 255.0f * temp[k]);
    }

    for (int j = 0; j < dst.rows; ++j) {
        for (int i = 0; i < dst.cols; ++i) {
            int temp_index = src.ptr<uchar>(j)[i];
            dst.at<uchar>(j, i) = (uchar) lut[temp_index];
        }
    }
    return dst;
}


int MeanFilter_Gray(int a, int b)
{
    Mat image, meanRes;
    image = imread("image/lena.png", 0); // Read the file
    namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("原图", image);                // Show our image inside it.

    blur(image, meanRes, Size(a, b));			//均值滤波

    namedWindow("均值滤波", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("均值滤波", meanRes);                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
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
     // MeanFilter_Gray(3, 3);
     // MeanFilter_Gray(5, 5);
     // MeanFilter_Gray(9, 9);

     // 2、利用高斯模板平滑灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 3*3、5*5 和 9*9 尺寸的高斯模板平滑灰度图像
     // GaussianFilter_Gray(3, 3);
     // GaussianFilter_Gray(5, 5);
     // GaussianFilter_Gray(9, 9);

     // 3、利用 Laplacian、Robert、Sobel 模板锐化灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，分别利用 Laplacian、Robert、 Sobel 模板锐化灰度图像
     // Laplacian_Gray();
     // Robert_G();
     // Sobel();

     // 4、利用高提升滤波算法增强灰度图像。 具体内容：利用 OpenCV 对图像像素进行操作，设计高提升滤波算法增 强图像
     // test4();

     // 5、利用均值模板平滑彩色图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，利 用 3*3、5*5 和 9*9 尺寸的均值模板平滑彩色图像
     // MeanFilter_Color(3, 3);
     // MeanFilter_Color(5, 5);
     // MeanFilter_Color(9, 9);

     // 6、利用高斯模板平滑彩色图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，分 别利用 3*3、5*5 和 9*9 尺寸的高斯模板平滑彩色图像
     // GaussianFilter_Color(3, 3);
     // GaussianFilter_Color(5, 5);
     // GaussianFilter_Color(9, 9);

     // 7、利用 Laplacian、Robert、Sobel 模板锐化灰度图像。 具体内容：利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作，分别利用 Laplacian、Robert、Sobel 模板锐化彩色图像
     // Lap2();
     // Laplacian_Color();
     // Robert_RGB();
     // Sobel_Color();
}