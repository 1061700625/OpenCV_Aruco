//
// Created by sxf on 22-4-20.
//

#ifndef DEMO_ELLIPSE_H
#define DEMO_ELLIPSE_H


#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>

void FeaturePoint(cv::Mat &img, cv::Mat &img2);
void drawCross(cv::Mat &img, cv::Point2f point, cv::Scalar color, int size, int thickness);
void EDCircle(cv::Mat &img);



#endif //DEMO_ELLIPSE_H
