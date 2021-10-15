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
int main(){
    Size image_size;
    int cnt=0;

    for (int i = 0; i < 24; i++){
        image_ori = cv::imread("/home/apricity/VsCodePro/Calibrate/calib2/"+std::__cxx11::to_string(i).append("_orig.jpg"));
        if(!cnt){
            image_size.width=image_ori.cols;
            image_size.height=image_ori.rows;
        }
        cvtColor(image_ori,image_gray,COLOR_BGR2GRAY);
        points_n=findChessboardCorners(image_gray,board_size,points);
        if ( points_n && points.size() == board_n ) {
            cnt++;
            find4QuadCornerSubpix( image_gray, points, Size( 5, 5 ) );
            points_class.push_back(points);
            image_draw=image_ori.clone();
            drawChessboardCorners( image_draw, board_size, points, points_n );
            imshow( "corners", image_draw );
            waitKey( 50 );
        }else{
            std::cout<<"failed with picture "+i<<std::endl;
        }
        points.clear();
    }
    std::cout << cnt << " useful chess boards" << std::endl;
    
    for (int i = 0; i < cnt; i++ ) {
        for (int j = 0; j < board_y; j++ ) {
            for (int k = 0; k < board_x; k++ ){
                Point3f pt;
                pt.x = k * cell_size.width;
                pt.y = j * cell_size.height;
                pt.z = 0;
                world_points.push_back( pt );
            }
        }
        world_points_class.push_back( world_points );
        world_points.clear();
    }

    Mat camera_matrix( 3, 3, CV_32FC1, Scalar::all( 0 ) );
    Mat dist_coeffs( 1, 5, CV_32FC1, Scalar::all( 0 ) );
    std::vector< Mat > rvecs;
    std::vector< Mat > tvecs;

    std::cout << calibrateCamera( world_points_class, points_class, image_size, camera_matrix, dist_coeffs, rvecs, tvecs ) << std::endl;
    std::cout << "camera_matrix: " <<camera_matrix << std::endl << "dist_coeffs: " << dist_coeffs << std::endl;
    return 0;
}