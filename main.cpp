#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "ellipse.h"
#include "aamed/FLED.h"



using namespace std;
using namespace cv;
using namespace dnn;

void siftTest() {
    int64 t1, t2;
    double tkpt, tdes, tmatch_bf, tmatch_knn;

    // 1. 读取图片
    const cv::Mat image1_color = cv::imread("../../images/1.2.png", cv::IMREAD_COLOR); //Load as grayscale
    const cv::Mat image2_color = cv::imread("../../images/2.png", cv::IMREAD_COLOR); //Load as grayscale
    cv::Mat image1, image2;
    cv::cvtColor(image1_color,image1,cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2_color,image2,cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;

    bool binaryMatch = false;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // 2. 计算特征点
    t1 = cv::getTickCount();
    sift->detect(image1, keypoints1);
    sift->detect(image2, keypoints2);
    t2 = cv::getTickCount();
    tkpt = 1000.0*(t2-t1) / cv::getTickFrequency();

    // 3. 计算特征描述符
    cv::Mat descriptors1, descriptors2;
    t1 = cv::getTickCount();
    sift->compute(image1, keypoints1, descriptors1);
    sift->compute(image2, keypoints2, descriptors2);
    t2 = cv::getTickCount();
    tdes = 1000.0*(t2-t1) / cv::getTickFrequency();

    // 4. 特征匹配
    auto desMatch = binaryMatch?(cv::DescriptorMatcher::BRUTEFORCE_HAMMING) : (cv::DescriptorMatcher::BRUTEFORCE);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(desMatch);

    // (1) 直接暴力匹配
    std::vector<cv::DMatch> matches;
    t1 = cv::getTickCount();
    matcher->match(descriptors1, descriptors2, matches);
    t2 = cv::getTickCount();
    tmatch_bf = 1000.0*(t2-t1) / cv::getTickFrequency();
    // 画匹配图
    cv::Mat img_matches_bf;
    drawMatches(image1_color, keypoints1, image2_color, keypoints2, matches, img_matches_bf);
    imshow("bf_matches", img_matches_bf);

    // (2) KNN-NNDR匹配法
    std::vector<std::vector<cv::DMatch> > knn_matches;
    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    t1 = cv::getTickCount();
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2);
    for (auto & knn_matche : knn_matches) {
        if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
            good_matches.push_back(knn_matche[0]);
        }
    }
    t2 = cv::getTickCount();
    tmatch_knn = 1000.0*(t2-t1) / cv::getTickFrequency();

    // 画匹配图
    cv::Mat img_matches_knn;
    drawMatches( image1_color, keypoints1, image2_color, keypoints2, good_matches, img_matches_knn, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::imshow("knn_matches", img_matches_knn);




    cv::Mat output;
    cv::drawKeypoints(image1_color, keypoints1, output);
    //cv::imwrite("sift_image1_keypoints.jpg", output);
    cv::imshow("sift_image1_keypoints", output);
    cv::drawKeypoints(image2_color, keypoints2, output);
    //cv::imwrite("sift_image2_keypoints.jpg", output);
    cv::imshow("sift_image2_keypoints", output);

    cv::waitKey(0);

    std::cout << "特征点检测耗时(ms)：" << tkpt << std::endl;
    std::cout << "特征描述符耗时(ms)：" << tdes << std::endl;
    std::cout << "BF特征匹配耗时(ms)：" << tmatch_bf << std::endl;
    std::cout << "KNN-NNDR特征匹配耗时(ms)：" << tmatch_knn << std::endl;
}

void fastrcnn() {
    const char* classNames[] = { "background", "person", "bicycle", "car",
                                 "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                                 "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep" };
    // 1. 读取图片
    const cv::Mat image1_color = cv::imread("../../images/1.png", cv::IMREAD_COLOR); //Load as grayscale
    const cv::Mat image2_color = cv::imread("../../images/2.png", cv::IMREAD_COLOR); //Load as grayscale
    cv::Mat image1, image2;
    cv::cvtColor(image1_color,image1,cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2_color,image2,cv::COLOR_BGR2GRAY);

    std::string pb_file = "/home/sxf/Desktop/RelativeLocation/models/frozen_inference_graph.pb";
    std::string pbtext_file = "/home/sxf/Desktop/RelativeLocation/models/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    auto net = cv::dnn::readNetFromTensorflow(pb_file, pbtext_file);
    float score_threshold = 0.3;

    double start = (double)getTickCount();
    net.setInput(cv::dnn::blobFromImage(image1_color,
                                        1.0,
                                        cv::Size(800, 600),
                                        cv::Scalar(),
                                        true,
                                        false));
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    auto output = net.forward();
    cout << "output size: " << output.size << endl;

    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    float confidenceThreshold = 0.50;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * image1_color.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * image1_color.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * image1_color.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * image1_color.rows);

            string conf = to_string(confidence);
            // sprintf(conf, "%0.2f", confidence);

            Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));
            rectangle(image1_color, object, Scalar(255, 0, 255), 2);
            // String label = String(classNames[objectClass]) + ": " + conf;
            String label = String("confidence: " + conf);
            int baseLine = 0;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseLine);
            rectangle(image1_color, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                                         Size(labelSize.width, labelSize.height + baseLine)),
                      Scalar(0, 255, 255), -1);
            putText(image1_color, label, Point(xLeftBottom, yLeftBottom),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
        }
    }
    double end = (double)getTickCount();
    cout << "use_time :" << (end - start) * 1000.0 / cv::getTickFrequency() << " ms \n";

    imshow("OpenCV DNN Test", image1_color);
    imwrite("result.jpg", image1_color);
    waitKey(0);
}

void AAMED_Fled(Mat &imgC, Mat &result, vector<Mat> &results) {
    Mat imgG;
    auto rows = imgC.size[0];
    auto cols = imgC.size[1];
    AAMED aamed(rows, cols);
    aamed.SetParameters(CV_PI / 3, 3.4, 0.77); // config parameters of AAMED
    cv::cvtColor(imgC, imgG, cv::COLOR_RGB2GRAY);
    aamed.run_FLED(imgG); // run AAMED

    cv::Vec<double, 10> detailTime;
    aamed.showDetailBreakdown(detailTime, true); // show detail execution time of each step
    aamed.drawFLED(imgG, "");
    // aamed.writeDetailData();

    cv::RotatedRect temp;
    Mat mask = Mat::zeros(imgC.size(), CV_8UC1);
    for (int i = 0; i < aamed.detEllipses.size(); i++) {
        Mat mask_temp = Mat::zeros(imgC.size(), CV_8UC1);
        Mat result_temp;
        temp.center.x = aamed.detEllipses[i].center.y;
        temp.center.y = aamed.detEllipses[i].center.x;
        temp.size.height = aamed.detEllipses[i].size.width+2;
        temp.size.width = aamed.detEllipses[i].size.height+2;
        temp.angle = -aamed.detEllipses[i].angle;
        // 以下部分是将几个区域存储在一起
        cv::ellipse(mask, temp, cv::Scalar(255), 2);
        cv::floodFill(mask, Point(temp.center.x, temp.center.y), Scalar(255));  // 填充背景为白色

        // 以下部分是将几个区域分别存储
        cv::ellipse(mask_temp, temp, cv::Scalar(255), 2);
        cv::floodFill(mask_temp, Point(temp.center.x, temp.center.y), Scalar(255));  // 填充背景为白色
        imgC.copyTo(result_temp, mask_temp);  // 找出ROI区域
        results.emplace_back(result_temp);
    }
    imgC.copyTo(result, mask);  // 找出ROI区域

    imshow("ellipse mask", mask);
    imshow("ellipse crop", result);
}

vector<Mat> generateAruco(int nums, int pixSize=200, int border=1) {
    vector<Mat> markerImages;
    Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    for (int i = 0; i < nums; ++i) {
        Mat tempImage;
        aruco::drawMarker(dictionary, i, pixSize, tempImage, border);
        markerImages.push_back(tempImage);
        imwrite("../markerImages"+to_string(i)+".png", tempImage);
    }

    return markerImages;
}

void detectAruco(const Mat& markerImage, vector<vector<Point2f>> &markerCorners, vector<vector<Point2f>> &rejectedCandidates, vector<int> &markerIds, Ptr<cv::aruco::Dictionary> dictionary, cv::Ptr<cv::aruco::GridBoard> board) {
    // 使用默认值初始化检测器参数
    Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    // 检测标记
    cv::aruco::detectMarkers(markerImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
    // 精细化再检测
    cv::aruco::refineDetectedMarkers(markerImage, board, markerCorners, markerIds, rejectedCandidates);
}

void estimatePose(vector<vector<Point2f>> &markerCorners, vector<int> &markerIds) {

    //cv::aruco::estimatePoseBoard(markerCorners, markerIds, board, cameraMatrix, distCoeffs, rvec, tvec);
}

int main(int argc, char* argv[])
{
    // 内参不知道哪个老哥整出来的
    double fx, fy, cx, cy, k1, k2, k3, p1, p2;
    fx = 604.5184;
    fy = 609.43059;
    cx = 397.4317;
    cy = 298.7116 ;
    k1 = 0.0406;
    k2 = -0.0774;
    k3 = 0;
    p1 = 1.8049e-4;
    p2 = 0;
    // 内参矩阵
    cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) <<
                                                    fx, 0.0, cx,
                                                    0.0, fy, cy,
                                                    0.0, 0.0, 1.0);
    // 畸变矩阵
    cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << k1, k2, p1, p2, k3);
    /*************************************预处理******************************************************/
    // 加载用于生成标记的字典。
    Ptr<cv::aruco::Dictionary> dictionary = getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    // board对象指针，在后面有create函数来实际创建
    // 下面这些参数需要用来计算相机位姿
    cv::Ptr<cv::aruco::GridBoard> board =  cv::aruco::GridBoard::create(
            2,             //每行多少个Marker
            2,             //每列多少个Marker
            0.04,          //marker长度, m
            0.01,           //marker之间的间隔, m
            dictionary);   //字典
    Mat boardImage;
    board->draw( cv::Size(200, 200),  // 整个board的大小
                 boardImage,                         // 返回的图像
                 10,                            // 整个board的边距
                 1 );                           // 每个码内的边距
    imwrite("../boardImage.png", boardImage);

    //vector<Mat> markerImages = generateAruco(5);
    //Mat markerImage = markerImages[0];
    Mat markerImage = imread("../1.png", 1);
    Mat markerImageCopy;
    markerImage.copyTo(markerImageCopy);

    cout<<">> 预处理完成!"<<endl;
    /************************************************************************************************/

    /*************************************检测椭圆*****************************************************/
    // FeaturePoint(img ,img2);
    //EDCircle(markerImage);
    vector<Mat> results;
    Mat fullSplitImage;
    AAMED_Fled(markerImage, fullSplitImage, results);
    /************************************************************************************************/

    // 对每个椭圆区域进行检测
    for (const auto& cropSplitImage: results) {
        //　检测Aruco
        vector<vector<Point2f>> markerCorners;
        vector<vector<Point2f>> rejectedCandidates;
        vector<int> markerIds;
        // cv::copyMakeBorder(markerImage, markerImage, 5, 5, 5, 5, cv::BORDER_CONSTANT, Scalar(255,0,0));
        detectAruco(cropSplitImage, markerCorners, rejectedCandidates, markerIds, dictionary, board);
        if (markerCorners.empty()) {
            cout<<"无可用Marker"<<endl;
            return 0;
        }

        // 显示检测到的但是由于字典对不上被拒绝的Marker
        if (!rejectedCandidates.empty()){
            cout<<"一共有 "<<rejectedCandidates.size()<<" 个被拒绝的 Marker "<<endl;
            for (auto & rejectedCandidate : rejectedCandidates) {
                for (int i=0;i<4;i++) {
                    cv::circle(fullSplitImage,cv::Point(rejectedCandidate[i].x,rejectedCandidate[i].y),6,cv::Scalar(0,0,255));
                }
            }
        }

        // 显示正确的Marker
        if (!markerIds.empty()) {
            // 绘制检测边框
            cv::aruco::drawDetectedMarkers(fullSplitImage, markerCorners, markerIds);

            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::Vec3d rvec, tvec;
            // 估计相机位姿(相对于每一个marker)
            //cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.055, cameraMatrix, distCoeffs, rvecs, tvecs);
            // 估计相机位姿(相对于 aruco 板)
            cv::aruco::estimatePoseBoard(markerCorners, markerIds, board, cameraMatrix, distCoeffs, rvec, tvec); rvecs.emplace_back(rvec); tvecs.emplace_back(tvec);
            // 为每个标记画轴
            for (int i = 0; i < rvecs.size(); ++i) {
                rvec = rvecs[i];
                tvec = tvecs[i];
                // 得到的位姿估计是：从board坐标系到相机坐标系的
                cv::Mat R;
                cv::Rodrigues(rvec,R);
                cout << "R_{camera<---marker} :" << R << endl;
                cout << "t_{camera<---marker} :" << tvec << endl;
                cout << endl;
                cv::aruco::drawAxis(fullSplitImage, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
                cv::aruco::drawAxis(markerImage, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
            }
        }
    }

    imshow("markerImage", markerImage);
    imshow("fullSplitImage", fullSplitImage);

    // imwrite("../markerImage.png", markerImage);
    waitKey(0);
    return 0;
}


