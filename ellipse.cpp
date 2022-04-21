//
// Created by sxf on 22-4-20.
//

#include "ellipse.h"
#include "ED_Lib/EDLib.h"

using namespace std;
using namespace cv;


void drawCross(Mat &img, Point2f point, Scalar color, int size, int thickness = 1)
{
    //绘制横线
    line(img, cvPoint(point.x - size / 2, point.y), cvPoint(point.x + size / 2, point.y), color, thickness, 8, 0);
    //绘制竖线
    line(img, cvPoint(point.x, point.y - size / 2), cvPoint(point.x, point.y + size / 2), color, thickness, 8, 0);
    return;
}


//提取单幅图像的特征点
void FeaturePoint(Mat &img, Mat &img2)
{
    //threshold(img, img, 40, 255, CV_THRESH_BINARY_INV);
    //将图片换为其它颜色
    Point2f center; //定义变量
    Point2f center1; //定义变量
    Mat edges;
    GaussianBlur(img, edges, Size(5, 5), 0, 0);
    Canny(edges, edges,35, 70, 3);
    //imshow("edges", edges);
    //waitKey(0);
    Mat mm = img2(Rect(0, 0, img2.cols , img2.rows));//ROI设置
    mm = { Scalar(0, 0, 0) };//把ROI中的像素值改为黑色
    vector<vector<Point> > contours;// 创建容器，存储轮廓
    vector<Vec4i> hierarchy;// 寻找轮廓所需参数

    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    Mat imageContours = Mat::zeros(edges.size(), CV_8UC1);//轮廓
    for (int i = 0; i < contours.size(); i++) {
        //绘制轮廓
        drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
    }

    //过滤
    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() <= 100 || contours[i].size() >= 1000){
            continue;
        }

        if (contourArea(contours[i]) < 10 || contourArea(contours[i]) > 40000) {
            continue;
        }
        //利用直线斜率处处相等的原理
        /*if (abs(((double)(contours[i][0].y - contours[i][20].y) / (double)(contours[i][0].x - contours[i][20].x) -
            (double)(contours[i][20].y - contours[i][40].y) / (double)(contours[i][20].x - contours[i][40].x)))<0.2)
                continue;*/
        //利用凹凸性的原理
        if (!abs((contours[i][0].y + contours[i][80].y) / 2 - contours[i][40].y))
            continue;

        RotatedRect m_ellipsetemp;  // fitEllipse返回值的数据类型
        m_ellipsetemp = fitEllipse(contours[i]);  //找到的第一个轮廓，放置到m_ellipsetemp
        if (m_ellipsetemp.size.width / m_ellipsetemp.size.height < 0.3) {
            continue;
        }
        cout << "1：" << contourArea(contours[i]) << endl;
        cout << "2：" << contours[i].size() << endl;
        //drawCross(img, center, Scalar(255, 0, 0), 30, 2);
        //ellipse(img, m_ellipsetemp, cv::Scalar(255, 255, 255));   //在图像中绘制椭圆
        ellipse(mm, m_ellipsetemp, cv::Scalar(255, 255, 255), FILLED);   //在图像中绘制椭圆
    }

    Mat edges1;
    GaussianBlur(mm, edges1, Size(5, 5), 0, 0);
    Canny(edges1, edges1, 50, 150, 3);
    //vector<vector<Point> > contours1;// 创建容器，存储轮廓
    //vector<Vec4i> hierarchy1;// 寻找轮廓所需参数
    findContours(edges1, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    Mat imageContours1 = Mat::zeros(edges1.size(), CV_8UC1);//轮廓

    for (int i = 0; i < contours.size(); i++) {
        drawContours(imageContours1, contours, i, Scalar(255), 1, 8, hierarchy);
        //绘制轮廓
    }
    for (int i = 0; i < contours.size(); i++)
    {
        if (contours[i].size() <= 100 || contours[i].size() >= 1000){
            continue;
        }
        if (contourArea(contours[i]) < 10 || contourArea(contours[i]) > 40000){
            continue;
        }
        RotatedRect m_ellipsetemp1;  // fitEllipse返回值的数据类型
        m_ellipsetemp1 = fitEllipse(contours[i]);  //找到的第一个轮廓，放置到m_ellipsetemp
        ellipse(img, m_ellipsetemp1, cv::Scalar(255, 0, 0));   //在图像中绘制椭圆
        cout << "面积：" << contourArea(contours[i]) << endl;
        cout << "点数：" << contours[i].size() << endl;
        center1 = m_ellipsetemp1.center;//读取椭圆中心
        drawCross(img, center1, Scalar(255, 0, 0), 30, 2);
        cout << center1.x << ", " << center1.y << endl;
    }
    //imshow("image", img);
    //imshow("imageContours1", imageContours1);
    //waitKey(0);
    //return ;//返回椭圆中心坐标
}

void EDCircle(Mat &img) {
    //*********************** EDCOLOR Edge Segment Detection from Color Images **********************
    EDColor testEDColor = EDColor(img, 36, 4, 1.5, true); //last parameter for validation
    imshow("Color Edge Image - PRESS ANY KEY TO QUIT", testEDColor.getEdgeImage());
    cout << "Number of edge segments detected by EDColor: " << testEDColor.getSegmentNo() << endl;

    // get lines from color image
    EDLines colorLine = EDLines(testEDColor);
    imshow("Color Line", colorLine.getLineImage());
    std::cout << "Number of line segments: " << colorLine.getLinesNo() << std::endl;

    // get circles from color image
    EDCircles colorCircle = EDCircles(testEDColor);
    // TO DO :: drawResult doesnt overlay (onImage = true) when input is from EDColor
    Mat circleImg = colorCircle.drawResult(false, ImageStyle::BOTH);
    imshow("Color Circle", circleImg);
    std::cout << "Number of line segments: " << colorCircle.getCirclesNo() << std::endl;
    waitKey();
}


















