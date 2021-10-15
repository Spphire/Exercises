#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <cmath>
using namespace std;
using namespace cv;

int main() {
    if(1){
        Mat OriginalImg0;
        OriginalImg0=imread("/home/apricity/VsCodePro/test2/apple.png");
        Mat OriginalImg;
        cvtColor(OriginalImg0,OriginalImg,COLOR_BGR2HSV);
        //threshold(OriginalImg,OriginalImg,90,255,THRESH_OTSU);//adaptive method
        Mat dst1;
        Mat dst2;
        Mat dst3;
        Mat dst4;
        Mat dst;
        inRange(OriginalImg,Scalar(0,150,150),Scalar(10,255,255),dst1);
        inRange(OriginalImg,Scalar(156,43,46),Scalar(180,255,255),dst2);
        inRange(OriginalImg,Scalar(15,210,86),Scalar(21,255,110),dst3);
        inRange(OriginalImg,Scalar(15,210,90),Scalar(21,255,110),dst4);
        imshow("dst1",dst1);
        imshow("dst2",dst2);
        imshow("dst3",dst3);
        imshow("dst4",dst4);
        dst=dst1+dst2+dst3;
        Mat element = getStructuringElement(MORPH_RECT, Size(11, 11));
        imshow("dst'",dst);
        morphologyEx(dst,dst, MORPH_CLOSE, element);
        morphologyEx(dst,dst, MORPH_OPEN, element);
        morphologyEx(dst,dst, MORPH_DILATE, element);
        imshow("dst",dst);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarcy;
        findContours(dst,contours,hierarcy,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<Rect> boundRect(contours.size());  //定义外接矩形集合
        Point2f rect[4];
        for (int i = 0; i < contours.size(); i++) {

            boundRect[i] = boundingRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
            rectangle(OriginalImg0, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);

        }
        imshow("final",OriginalImg0);
        waitKey(0);
        return 0;
    }
}
