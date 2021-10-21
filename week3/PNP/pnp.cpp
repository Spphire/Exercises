#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

const int board_x=9,board_y=6;
const int board_n=board_x*board_y;
Size board_size(board_x,board_y);
Size cell_size(10,10);
Mat image_ori,image_gray, image_draw;
std::vector< Point2f > points;
int points_n;
std::vector< std::vector<Point2f> > points_class;

std::vector< Point3f > world_points;
std::vector< std::vector<Point3f> > world_points_class;

Mat camera_matrix= (Mat_<double>(3,3) <<  9.1234465679824348e+02, 0., 6.4157634413436961e+02, 0.,
       7.6573154962089018e+02, 3.6477945186797331e+02, 0., 0., 1. );
Mat dist_coeffs= (Mat_<double>(1,5) <<  0., -4.2669718747763807e-01, 2.6509688616309912e-01,
       -5.3624979910268683e-04, -4.1011485564833132e-04  );
Mat rvecs;
Mat tvecs;
int main(){
    Size image_size;
    int cnt=0;

    
    for (int j = 0; j < board_y; j++ ) {
        for (int k = 0; k < board_x; k++ ){
            Point3f pt;
            pt.x = k * cell_size.width;
            pt.y = j * cell_size.height;
            pt.z = 0;
            world_points.push_back( pt );
        }
    }
    
    


    for (int i = 0; i < 41; i++){
        image_ori = cv::imread("../chess/"+std::__cxx11::to_string(i).append(".jpg"));
        if(!cnt){
            image_size.width=image_ori.cols;
            image_size.height=image_ori.rows;
        }
        cvtColor(image_ori,image_gray,COLOR_BGR2GRAY);
        points_n=findChessboardCorners(image_gray,board_size,points);
        if ( points_n && points.size() == board_n ) {
            cnt++;
            find4QuadCornerSubpix( image_gray, points, Size( 5, 5 ) );
            solvePnP(world_points,points,camera_matrix,dist_coeffs,rvecs,tvecs);
            std::cout<<"image:"<<i<<std::endl;
            std::cout<<"rvec: "<<rvecs<<std::endl;
            std::cout<<"tvec: "<<tvecs<<std::endl;
            points_class.push_back(points);
            image_draw=image_ori.clone();
            drawChessboardCorners( image_draw, board_size, points, points_n );
            imshow( "corners", image_draw );
            int key=waitKey( 0 );
            if(char(key)=='q'){
                break;
            }
        }else{
            std::cout<<"failed with picture "+i<<std::endl;
        }
        points.clear();
    }
    
    
    

    
    
    return 0;
}