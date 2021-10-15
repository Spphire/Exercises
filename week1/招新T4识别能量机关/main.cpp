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
#define Max(a,b) ((a>b)?a:b)
#define Min(a,b) ((a<b)?a:b)
cv::Point2f ctp;
double getDistance(cv::Point2f p1,cv::Point2f p2){
    return	sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}
bool cmp(cv::Point2f p1,cv::Point2f p2){
    return getDistance(p1,ctp) < getDistance(p2,ctp);
}
cv::Point2f get5thPoint(cv::Point2f P[],cv::Point2f Rpoint){
    ctp = Rpoint;
    sort(P,P+4,cmp);
    return cv::Point2f((P[0].x+P[1].x)/2, (P[0].y+P[1].y)/2);
}
int getRate(cv::Point2f P[],cv::Point2f Rpoint){
    int min1 =0;
    int min2 =1;
    for(int i=2;i<7;i++){
        if(getDistance(P[i%4],Rpoint)<getDistance(P[min1],Rpoint)) {
            if (min2 != i % 4) {
                min1 = i % 4;
            } else {
                int tt = min2;
                min2 = min1;
                min1 = tt;
            }
        }else if(getDistance(P[i%4],Rpoint)<getDistance(P[min2],Rpoint)){
            if(min1!=i%4) {
                min2 = i % 4;
            }
        }
    }
    if(Max(min1,min2)!=3){
        return max(min1,min2);
    }else if(Min(min1,min2)==0){
        return 0;
    }else{
        return max(min1,min2);
    }

}

cv::Point2f _get5thPoint(cv::Point2f P[],cv::Point2f Rpoint){
    ctp = Rpoint;
    sort(P,P+4,cmp);
    double x1 = P[0].x,y1 = P[0].y;
    double x2 = P[1].x,y2 = P[1].y;
    double x3 = (P[2].x+P[3].x)/2,y3 = (P[2].y+P[3].y)/2;
    double x4 = Rpoint.x,y4 = Rpoint.y;
    double resX = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4));
    double resY = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4));
    //return cv::Point2f((P[0].x+P[1].x)/2, (P[0].y+P[1].y)/2);
    return cv::Point2f(resX,resY);
}

int main() {
    const int cycR=20;//to find out centre

    VideoCapture capture;
    Mat OriginalImg;
    capture.open("/home/apricity/CLionProjects/Test/video001.avi");
    if(!capture.isOpened()){
        cout<<"can not open ..."<<endl;
        return-1;
    }
    Point2f centre;
    while(capture.read(OriginalImg)) {
        //capture>>OriginalImg;


        //OriginalImg = imread("../ai3.jpg", IMREAD_COLOR);//读取原始彩色图像
        if (OriginalImg.empty())  //判断图像对否读取成功
        {
            cout << "错误!读取图像失败\n";
            return -1;
        }
        imshow("begin", OriginalImg);

        Mat srcImage = OriginalImg;
        vector<Mat> channels;    //vector<Mat>： 可以理解为存放Mat类型的容器（数组）
        split(srcImage, channels);  //对原图像进行通道分离，即把一个3通道图像转换成为3个单通道图像channels[0],channels[1] ,channels[2]
        vector<Mat> mbgr(3);    //创建类型为Mat，数组长度为3的变量mbgr
        Mat hideChannel(srcImage.size(), CV_8UC1, Scalar(0));//需要隐藏的通道。尺寸与srcImage相同，单通道黑色图像。
        //Apricity-
        //注意：0通道为B分量，1通道为G分量，2通道为R分量。因为：RGB色彩空间在opencv中默认通道顺序为BGR！！！
        //【1】显示彩色的B-蓝色分量。
        Mat imageB(srcImage.size(), CV_8UC3);    //创建尺寸与srcImage相同，三通道图像imageB
        mbgr[0] = channels[0];
        mbgr[1] = hideChannel;
        mbgr[2] = hideChannel;
        merge(mbgr, imageB);
        //imshow("imageB-蓝色通道", imageB);

        //【2】显示彩色的G分量
        Mat imageG(srcImage.size(), CV_8UC3);//创建尺寸与srcImage相同，三通道图像imageG
        mbgr[0] = hideChannel;
        mbgr[1] = channels[1];
        mbgr[2] = hideChannel;
        merge(mbgr, imageG);
        //imshow("imageG-绿色通道", imageG);

        //【3】显示彩色的R分量
        Mat imageR(srcImage.size(), CV_8UC3);//创建尺寸与srcImage相同，三通道图像imageR
        mbgr[0] = hideChannel;
        mbgr[1] = hideChannel;
        mbgr[2] = channels[2];
        merge(mbgr, imageR);
        //imshow("imageR-红色通道", imageR);

        Mat B = imageB.clone();

        blur(imageB, imageB, Size(3, 3));
        cvtColor(srcImage, srcImage, COLOR_RGB2GRAY);
        cvtColor(imageB, imageB, COLOR_RGB2GRAY);

        srcImage *= 1.5;
        srcImage *= 3;
        srcImage -= 20;
        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(srcImage, srcImage, MORPH_OPEN, element);

        imageB = (imageB * 11 - 140) * 1.2;
        imageB *= 4;
        blur(imageB, imageB, Size(3, 3));

        Mat dst = srcImage - imageB;

        morphologyEx(dst, dst, MORPH_OPEN, element);
        namedWindow("形态学处理操作", WINDOW_NORMAL);

        dst = dst * 8 - 700;


        Mat e = getStructuringElement(MORPH_RECT, Size(6, 6));
        morphologyEx(dst, dst, MORPH_CLOSE, e);
        imshow("binery", dst);

        Mat candy_img;
        Canny(dst, candy_img, 300, 200, 3);
        imshow("canny", candy_img);


        Mat srcImg = dst.clone();
        Mat dstImg = OriginalImg.clone();


        vector<vector<Point>> contours;
        vector<Vec4i> hierarcy;
        vector<vector<Point>> contoursInside;
        vector<Vec4i> hierarcyInside;


        findContours(srcImg, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<Rect> boundRect(contours.size());  //定义外接矩形集合

        findContours(srcImg, contoursInside, hierarcyInside, RETR_CCOMP, CHAIN_APPROX_NONE);
        vector<Rect> inboundRect(contoursInside.size());  //定义in接矩形集合

        vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
        vector<RotatedRect> boxIn(contoursInside.size()); //定义最小in接矩形集合

        vector<vector<Point>> appro(contours.size());//approxPoly
        Point2f rect[4];


        Point2f rectIn[4];
//Inside
        cout << contours.size() << endl;

        vector<vector<Point>> endline;
        vector<int> Blue;
        vector<int> Green;

//GET CENTER-R

        for (int i = 0; i < contours.size(); i++) {

            box[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
            box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
            if (Max(getDistance(rect[0], rect[2]), getDistance(rect[1], rect[3])) < 8) {
                continue;
            }
            //circle(dstImg, rect[0], 3, Scalar(255, 255, 255), -1, 8);
            //circle(dstImg, Point(box[i].center.x, box[i].center.y), 3, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
            //rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);

            double area = contourArea(contours[i], false);
            cout<<"centre area: "<<area<<endl;
            if (50 < area && area < 500) {
                string flag;
                if(dst.at<uchar>(box[i].center)<100){ flag="true";}else{flag="false";}
                cout<<"bin: "<<flag<<endl;
                if((OriginalImg.at<Vec3b>(box[i].center)[1]>60&&OriginalImg.at<Vec3b>(box[i].center)[2]<100)||(dst.at<uchar>(box[i].center)<100)) {
                    continue;
                }
                centre = Point(box[i].center.x, box[i].center.y);

            }else if(area>1000){

            }

        }
        circle(dstImg, centre,Max(Min(Max(getDistance(rect[0], rect[2]), getDistance(rect[1], rect[3])) / 2,10),9), Scalar(255, 255, 255), 1,8);
        circle(dstImg, centre,2, Scalar(255, 255, 255), -1,8);
        //GET CENTER-R
        cout<<"centre: "<<centre.x<<","<<centre.y<<endl;
        cout<<"area"<<endl;
        //GET 4/5 +1/5
        for (int i = 0; i < contoursInside.size(); i++) {
            if (hierarcyInside[i][3] < 0) {
                continue;
            } else {
                boxIn[i] = minAreaRect(Mat(contoursInside[i]));
                boxIn[i].points(rectIn);

                if(OriginalImg.at<Vec3b>(boxIn[i].center)[1]>60&&OriginalImg.at<Vec3b>(boxIn[i].center)[2]<100) {
                    continue;
                }

                if (getDistance(rectIn[0], rectIn[2]) > 60) {
                    continue;
                }
                vector<Point> con;
                //circle(dstImg, rectIn[0], 5, Scalar(255, 255, 255), 3, 8);
                //circle(dstImg, rectIn[1], 5, Scalar(0, 0, 255), 3, 8);
                //circle(dstImg, rectIn[2], 5, Scalar(0, 255, 0), 3, 8);
                //circle(dstImg, rectIn[3], 5, Scalar(255, 0, 0), 3, 8);
                cout << "Rate: "<<getRate(rectIn, centre) << endl;
                int Rate0 = getRate(rectIn, centre);
                for (int j = 0; j < 4; j++) {
                    con.push_back(rectIn[(Rate0 + j) % 4]);

                }

                //-
                boxIn[i] = minAreaRect(Mat(contoursInside[hierarcyInside[i][3]]));  //计算每个轮廓最小外接矩形

                boxIn[i].points(rectIn);  //把最小外接矩形四个端点复制给rect数组
                //int u=3,v=5;
                //con.push_back(  (  u*(rectIn[Rate0]+rectIn[(Rate0+5)%4])  /2+v*centre)/(u+v)  );
                con.push_back(_get5thPoint(rectIn, centre));
                endline.push_back(con);
                double area = contourArea(contoursInside[hierarcyInside[i][3]], false);
                cout << area << endl;
                if (area > 4000) {
                    Blue.push_back(255);
                    Green.push_back(0);
                } else if (area < 500) {

                } else {
                    Blue.push_back(0);
                    Green.push_back(255);
                }
                //-
            }
        }
//GET 5/5
        cout << "finish geting" << endl;
//DRAW
        for (int i = 0; i < endline.size(); i++) {
            drawContours(dstImg, endline, i, Scalar(Blue[i], Green[i], 0), 2, 8);  //绘制多边形逼近
        }

        imshow("end", dstImg);

        waitKey(15);
    }
    capture.release();
    return 0;

}
