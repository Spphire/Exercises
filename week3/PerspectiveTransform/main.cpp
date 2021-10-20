#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

void doPerspectiveTransform( Mat input, Mat& output , std::vector<Point2f> srcQ) {
    std::vector<Point2f> srcQuad( 4 ), dstQuad( 4 ),delta(4);
    srcQ[0].x+=0;
    srcQ[0].y+=0;
    srcQ[1].x+=1;
    srcQ[1].y+=0;
    srcQ[2].x+=0;
    srcQ[2].y+=0;
    srcQ[3].x+=3;
    srcQ[3].y+=0;

    output = input.clone();
    srcQuad[0].x = 598, srcQuad[1].x =  605 , srcQuad[2].x = 772, srcQuad[3].x = 763;
    srcQuad[0].y = 331, srcQuad[1].y = 389, srcQuad[2].y = 389, srcQuad[3].y = 335;
    dstQuad[0].x = 0, dstQuad[1].x = 0, dstQuad[2].x = 600, dstQuad[3].x = 600;
    dstQuad[0].y = 0, dstQuad[1].y = 200, dstQuad[2].y = 200, dstQuad[3].y = 0;
    
    Mat warp_matrix = getPerspectiveTransform( srcQ, dstQuad );

    warpPerspective( input, output, warp_matrix, Size(600,200) );
}
std::vector<Point2f> Recognition(Mat input){
    Mat OriginalImg;
    cvtColor(input,OriginalImg,COLOR_BGR2HSV);
    //threshold(OriginalImg,OriginalImg,90,255,THRESH_OTSU);//adaptive method
    Mat dst1;
    //Mat dst2;
    //Mat dst3;
    //Mat dst4;
    Mat dst;
    inRange(OriginalImg,Scalar(103,113,102),Scalar(153,255,187),dst1);
    //inRange(OriginalImg,Scalar(94,145,102),Scalar(153,255,187),dst1);
    //inRange(OriginalImg,Scalar(97,100,123),Scalar(153,255,171),dst1);
    //inRange(OriginalImg,Scalar(156,38,44),Scalar(180,255,255),dst2);
    //inRange(OriginalImg,Scalar(15,210,86),Scalar(21,255,110),dst3);
    //inRange(OriginalImg,Scalar(15,210,90),Scalar(21,255,110),dst4);
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat e = getStructuringElement(MORPH_RECT, Size(5, 5));
    //morphologyEx(dst1,dst1, MORPH_CLOSE, e);
    //morphologyEx(dst1,dst1, MORPH_CLOSE, element);
    morphologyEx(dst1,dst1, MORPH_CLOSE, element);
    dst=dst1;

    imshow("dst",dst);
    
    std::vector<std::vector<cv::Point>> roi_point;
    findContours(dst,roi_point,RETR_EXTERNAL,CHAIN_APPROX_NONE);

    /*
    std::cerr<<roi_point[0]<<std::endl;
    for (int i = 0; i < roi_point.size(); i++) {
        drawContours(input, roi_point, i, Scalar(0, 255, 0), 1, 8);
    }
    std::cerr<<"2"<<std::endl;
    imshow("dstImg",input);
    */
    std::vector<Point2f> roi_point_approx;
    Mat roi_approx(dst.size(),CV_8UC3,Scalar(0,0,0));
    auto i = roi_point.begin();
    approxPolyDP( *i, roi_point_approx, 7, 1 );
    double k=1;
    for(auto a : roi_point_approx){
        circle(roi_approx,a,2,Scalar(0,0,k*25));
        k++;
    }
    imshow("roi_approx",roi_approx);


    return roi_point_approx;
    
}


int main() {
    Mat img = imread("../car.jpg");
    Mat result;
    //Recognition(img);
    doPerspectiveTransform( img, result ,Recognition(img));

    namedWindow( "input" );
    namedWindow( "output" );

    imshow( "input", img );
    imshow( "output", result );

    waitKey( 0 );

    destroyWindow( "input" );
    destroyWindow( "output" );
    return 0;
}