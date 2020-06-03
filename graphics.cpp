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


    lab2();

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

Mat myEqualizeHist2(Mat src, Mat dst) {
    // if (src.empty())
    // {
    //     return dst;
    // }
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat _src(src.size(), src.type());
    src.copyTo(_src);
    int hist[256] = {0,};
    int lut[256];
    cv::Range heightRange(0, _src.rows);

    const size_t sstep = _src.step;
    int width = _src.cols;
    int height = heightRange.end - heightRange.start;
    // if (_src.isContinuous())
    // {
    //     width *= height;
    //     height = 1;
    // }
    for (const uchar *ptr = _src.ptr<uchar>(heightRange.start); height--; ptr += sstep) {
        int x = 0;
        for (; x <= width - 4; x += 4) {
            int t0 = ptr[x], t1 = ptr[x + 1];
            hist[t0]++;
            hist[t1]++;
            t0 = ptr[x + 2], t1 = ptr[x + 3];
            hist[t0]++;
            hist[t1]++;
        }
        for (; x < width; ++x) {
            hist[ptr[x]]++;
        }
    }

    int i = 0;
    while (!hist[i]) ++i;

    int total = (int) _src.total();
    if (hist[i] == total) {
        dst.setTo(i);
        return dst;
    }
    float scale = (256 - 1.f) / (total - hist[i]);
    int sum = 0;
    for (lut[i++] = 0; i < 256; ++i) {
        sum += hist[i];
        lut[i] = cv::saturate_cast<uchar>(sum * scale);
    }

    cv::Range heightRange1(0, _src.rows);
    const size_t dstep = dst.step;
    int height1 = heightRange1.end - heightRange1.start;
    int width1 = _src.cols;
    // if (_src.isContinuous() && dst.isContinuous())
    // {
    //     width1 *= height1;
    //     height1 = 1;
    // }
    const uchar *sptr1 = _src.ptr<uchar>(heightRange1.start);
    uchar *dptr1 = dst.ptr<uchar>(heightRange1.start);
    for (; height1--; sptr1 += sstep, dptr1 += dstep) {
        int x = 0;
        for (; x <= width1 - 4; x += 4) {
            int v0 = sptr1[x];
            int v1 = sptr1[x + 1];
            int x0 = lut[v0];
            int x1 = lut[v1];
            dptr1[x] = (uchar) x0;
            dptr1[x + 1] = (uchar) x1;

            v0 = sptr1[x + 2];
            v1 = sptr1[x + 3];
            x0 = lut[v0];
            x1 = lut[v1];
            dptr1[x + 2] = (uchar) x0;
            dptr1[x + 3] = (uchar) x1;
        }
        for (; x < width1; ++x) {
            dptr1[x] = (uchar) lut[sptr1[x]];
        }
    }
    return dst;
}


Mat myEqualizeHist(Mat src, Mat dst) {
    // if (src.empty())
    // {
    //     return dst;
    // }
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat _src(src.size(), src.type());
    src.copyTo(_src);
    int hist[256] = {0,};
    int lut[256];
    cv::Range heightRange(0, _src.rows);

    const size_t sstep = _src.step;
    int width = _src.cols;
    int height = heightRange.end - heightRange.start;
    // if (_src.isContinuous())
    // {
    //     width *= height;
    //     height = 1;
    // }
    for (const uchar *ptr = _src.ptr<uchar>(heightRange.start); height--; ptr += sstep) {
        int x = 0;
        for (; x <= width - 4; x += 4) {
            int t0 = ptr[x], t1 = ptr[x + 1];
            hist[t0]++;
            hist[t1]++;
            t0 = ptr[x + 2], t1 = ptr[x + 3];
            hist[t0]++;
            hist[t1]++;
        }
        for (; x < width; ++x) {
            hist[ptr[x]]++;
        }
    }

    int i = 0;
    while (!hist[i]) ++i;

    int total = (int) _src.total();
    if (hist[i] == total) {
        dst.setTo(i);
        return dst;
    }
    float scale = (256 - 1.f) / (total - hist[i]);
    int sum = 0;
    for (lut[i++] = 0; i < 256; ++i) {
        sum += hist[i];
        lut[i] = cv::saturate_cast<uchar>(sum * scale);
    }

    cv::Range heightRange1(0, _src.rows);
    const size_t dstep = dst.step;
    int height1 = heightRange1.end - heightRange1.start;
    int width1 = _src.cols;
    // if (_src.isContinuous() && dst.isContinuous())
    // {
    //     width1 *= height1;
    //     height1 = 1;
    // }
    const uchar *sptr1 = _src.ptr<uchar>(heightRange1.start);
    uchar *dptr1 = dst.ptr<uchar>(heightRange1.start);
    for (; height1--; sptr1 += sstep, dptr1 += dstep) {
        int x = 0;
        for (; x <= width1 - 4; x += 4) {
            int v0 = sptr1[x];
            int v1 = sptr1[x + 1];
            int x0 = lut[v0];
            int x1 = lut[v1];
            dptr1[x] = (uchar) x0;
            dptr1[x + 1] = (uchar) x1;

            v0 = sptr1[x + 2];
            v1 = sptr1[x + 3];
            x0 = lut[v0];
            x1 = lut[v1];
            dptr1[x + 2] = (uchar) x0;
            dptr1[x + 3] = (uchar) x1;
        }
        for (; x < width1; ++x) {
            dptr1[x] = (uchar) lut[sptr1[x]];
        }
    }
    return dst;
}