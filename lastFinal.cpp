#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "LaneDetector.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <thread>
#include <math.h>
#include <iomanip>


#define VIDEO_FILE_NAME "video.mp4"
#define VIDEO_FILE_NAME1 "video1.avi"
#define VIDEO_FILE_NAME2 "video2.avi"
#define VIDEO_FILE_NAME3 "video3.avi"
#define VIDEO_FILE_NAME4 "video4.avi"
#define VIDEO_FILE_NAME5 "video5.mp4"
#define VIDEO_FILE_NAME6 "video6.avi"
#define VIDEO_FILE_NAME7 "video7.avi"
#define VIDEO_FILE_NAME8 "video8.avi"
#define VIDEO_FILE_NAME9 "video9.avi"


#define CASCADE_FILE_NAME1 "/home/pi/sys/cars.xml"
#define CASCADE_FILE_NAME2 "/home/pi/sys/cars2.xml"
#define CASCADE_FILE_NAME3 "/home/pi/sys/cars3.xml"
#define CASCADE_FILE_NAME4 "/home/pi/sys/cas2.xml"
#define CASCADE_FILE_NAME5 "/home/pi/sys/cas3.xml"
#define CASCADE_FILE_NAME6 "/home/pi/sys/cas4.xml"



#define CASCADE4_FILE_NAME "/home/pi/sys/left-sign.xml"
#define CASCADE5_FILE_NAME "/home/pi/sys/right-sign.xml"

#define CAR_IMAGE "/home/pi/sys/car.png"
#define LEFT_SIGN_IMAGE "/home/pi/sys/left.png"
#define RIGHT_SIGN_IMAGE "/home/pi/sys/right.png"

#define WINDOW_NAME_1 "WINDOW1"
#define WINDOW_NAME_2 "WINDOW2"

int lineRFlag = 1;
int lineLFlag = 1;
int LineCFlag = 1;
int sign60Flag = 1;
int sign70Flag = 1;
int sign80Flag = 1;
int sign90Flag = 1;

using namespace cv;
using namespace std;

void thread_start(int value );
void thread_function(int value);

void makeSqare (Mat mat,Mat roi,vector<Point2f> sceneP);
void kkazeF(Ptr<AKAZE> kazeF,Mat srcImage1,Mat roi,vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,Mat descriptors1,Mat descriptors2);
//void signMatching(vector<KeyPoint> *keypoints[4], Mat *descriptorses[4], Mat *signMachings[4]);


Point GetWrappedPoint(Mat M, const Point& p);
void draw_locations(Mat & img, vector< Rect > & locations, const Scalar & color,string text);

vector< DMatch > matching (vector< DMatch > goodMatches,Mat descriptors1,Mat descriptors2);
vector<Point2f> betweenPointAndPoint(Mat mat,vector<KeyPoint> keypoints1,vector<KeyPoint> keypoints2, vector< DMatch > goodMatches,vector<Point2f> sceneP);

/** Make video window **/
int makeWindow(VideoCapture capture, string s);

/** original size print **/
void printoriginal(VideoCapture capture);
 
 
cv::Mat LaneDetector::deNoise(cv::Mat original){ 
cv::Mat output;
cv::GaussianBlur(original, output, cv::Size(3, 3), 0, 0); // GaussianBlur 적용 
return output;
}

// 가장자리 감지
// 이미지를 필터링하여 흐린 프레임의 모든 가장자리를 감지합니다
// img_noise는 이전에 흐려진 프레임입니다
// 반환 흰색으로 표시된 가장자리 만있는 이진 이미지
cv::Mat LaneDetector::edgeDetector(cv::Mat img_noise) {
  cv::Mat output;
  cv::Mat kernel;
  cv::Point anchor;
  cv::Mat gaussianLine;
  cv::cvtColor(img_noise, gaussianLine, cv::COLOR_RGB2GRAY); // 이미지를 RGB에서 회색으로 변환
  cv::GaussianBlur(gaussianLine,output,cv::Size(3, 3), 1);
  cv::threshold(output, output, 150, 180, cv::THRESH_BINARY); // 회색 이미지 바이너리 화

  // 커널 만들기 [-1 0 1]
  // Mathworks의 차선 이탈 경고 시스템의 커널을 기반으로합니다.
  anchor = cv::Point(-1, -1);
  kernel = cv::Mat(1, 3, CV_32F);
  kernel.at<float>(0, 0) = -1;
  kernel.at<float>(0, 1) = 0;
  kernel.at<float>(0, 2) = 1;
  
  cv::filter2D(output, output, -1, kernel, anchor, 0, cv::BORDER_DEFAULT); // 이진 이미지를 필터링하여 가장자리를 얻음
  
  return output;
}

// 가장자리 이미지 마스킹
// 차선의 일부를 형성하는 가장자리 만 감지되도록 이미지를가립니다.
// img_edges는 이전 함수의 가장자리 이미지입니다.
// return 원하는 가장자리 만 표현 된 이진 이미지를 반환합니다.
cv::Mat LaneDetector::mask(cv::Mat img_edges) {
  cv::Mat output;
  cv::Mat mask = cv::Mat::zeros(img_edges.size(), img_edges.type());
  
  cv::Point pts[4]= {     // 마스크 좌표
    cv::Point(200, 650), // 250 680    210  720
      cv::Point(500, 500), // 550 500    550  450
      cv::Point(600, 500), // 700 500    720  450
      cv::Point(430, 650) // 1000 680   1280 720
     

};
 cv::Point pts2[4]= {     // 마스크 좌표
  
     
	 cv::Point(820,650),
	 cv:: Point(650,500),
	cv::Point(750,500),
	cv::Point(1050,650)  
};

 

  cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255, 0, 0)); // 이진 다각형 마스크 만들기
	cv::fillConvexPoly(mask, pts2, 4, cv::Scalar(255, 0, 0)); // 이진 다각형 마스크 만들기

//mshow("maskmask",mask);
  cv::bitwise_and(img_edges, mask, output); // 출력을 얻으려면 가장자리 이미지와 마스크를 곱함
 	

  return output;
}

// 허프 라인
// 레인 경계의 일부가 될 마스크 된 이미지의 모든 선분을 얻습니다.
// img_mask는 이전 함수의 마스크 된 이진 이미지입니다.
// 반환 이미지의 모든 감지 된 선을 포함하는 벡터
std::vector<cv::Vec4i> LaneDetector::houghLines(cv::Mat img_mask) {
 std::vector<cv::Vec4i> line;

 HoughLinesP(img_mask, line, 1 , CV_PI/180, 20, 20, 30); // 20 20 30 ρ와 theta는 시행 착오에 의해 선택된다.
							// 임계값은 클수록 더 직선인 것만 찾음 
							// 엣지 점 최대 허용 간격은 작을수록 더 직선에 가까운 선만 잡음
 return line;
}

// 직선 및 좌선 정렬
// 검출 된 모든 Hough 선을 기울기별로 정렬하십시오.
// 라인은 오른쪽 또는 왼쪽으로 분류됩니다.
// 사면의 표시와 대략적인 위치
// lines은 모든 감지 된 라인을 포함하는 벡터입니다.
// img_edges는 이미지 중심을 결정하는 데 사용됩니다.
// return 출력은 모든 분류 된 선을 포함하는 벡터 (2)입니다.
std::vector<std::vector<cv::Vec4i> > LaneDetector::lineSeparation(std::vector<cv::Vec4i> lines, cv::Mat img_edges) {
  std::vector<std::vector<cv::Vec4i> > output(2);
  size_t j = 0;
  cv::Point ini;
  cv::Point fini;
  double slope_thresh = 0.3;
  std::vector<double> slopes;
  std::vector<cv::Vec4i> selected_lines;
  std::vector<cv::Vec4i> right_lines, left_lines;

// 검출 된 모든 선의 기울기 계산
  for (auto i : lines) { 
    ini = cv::Point(i[0], i[1]);
    fini = cv::Point(i[2], i[3]);

    // 기본 대수 : 기울기(m) = (y1 - y0) / (x1 - x0)
    double slope = (static_cast<double>(fini.y) - static_cast<double>(ini.y))/(static_cast<double>(fini.x) - static_cast<double>(ini.x) + 0.00001);

    if (std::abs(slope) > slope_thresh) { // 경사가 너무 수평이 맞으면 선을 버립니다.
      slopes.push_back(slope);            // 그렇지 않은 경우에는 해당 경사면과 해당 경사면을 저장
      selected_lines.push_back(i);
    }
  }

  // 선을 오른쪽과 왼쪽 선으로 나눕니다.
  img_center = static_cast<double>((img_edges.cols / 2)); // 중심을 구함
  
  while (j < selected_lines.size()) {
    ini = cv::Point(selected_lines[j][0], selected_lines[j][1]);
    fini = cv::Point(selected_lines[j][2], selected_lines[j][3]);
  
    // 라인을 왼쪽 또는 오른쪽으로 분류하는 조건
    if (slopes[j] > 0 && fini.x > img_center && ini.x > img_center) {
      right_lines.push_back(selected_lines[j]);
      right_flag = true;
    } else if (slopes[j] < 0 && fini.x < img_center && ini.x < img_center) {
      left_lines.push_back(selected_lines[j]);
      left_flag = true;
    } 
    j++;
  }
  
  output[0] = right_lines;
  output[1] = left_lines;

  return output;
}

// 왼쪽 및 오른쪽 라인에 대한 회귀
// 요약 회귀 분석은 분류 된 모든 선분을 초기 및 최종 점으로 취하여 최소 제곱 법을 사용하여 새 선을 그 안에 맞 춥니 다.
// 이것은 왼쪽과 오른쪽의 양면에 대해 수행됩니다.
// left_right_lines는 lineSeparation 함수의 출력입니다.
// original는, 행의 종료 위치를 선택하기 위해서 사용됩니다.
// return 출력은 두 차선 경계선의 초기 및 최종 점을 포함합니다.

std::vector<cv::Point> LaneDetector::regression(std::vector<std::vector<cv::Vec4i>> left_right_center_lines, cv::Mat original) {
  std::vector<cv::Point> output(6);
  cv::Point ini;
  cv::Point fini;
  cv::Point ini2;
  cv::Point fini2;

  cv::Vec4d right_line;
  cv::Vec4d left_line;

  std::vector<cv::Point> right_pts;
  std::vector<cv::Point> left_pts;


  // 올바른 라인이 감지되면, 라인의 모든 시작점과 마지막 점을 사용하여 라인을 맞 춥니다.
  if (right_flag == true) {
    for (auto i : left_right_center_lines[0]) {
      ini = cv::Point(i[0], i[1]);
      fini = cv::Point(i[2], i[3]);
      right_pts.push_back(ini);
      right_pts.push_back(fini);
    }

    if (right_pts.size() > 0) {
      //오른쪽 선이 여기에 형성됩니다.
      cv::fitLine(right_pts, right_line, CV_DIST_L2, 0, 0.01, 0.01); // 0.01 0.01
      right_m = right_line[1] / right_line[0];
      right_b = cv::Point(right_line[2], right_line[3]);
    }
  }

  // 왼쪽 라인이 감지되면, 라인의 모든 init 및 최종 포인트를 사용하여 라인을 맞추십시오
  if (left_flag == true) {
    for (auto j : left_right_center_lines[1]) {
      ini2 = cv::Point(j[0], j[1]); 
      fini2 = cv::Point(j[2], j[3]);
      left_pts.push_back(ini2);
      left_pts.push_back(fini2);
    }

    if (left_pts.size() > 0) {
      // 왼쪽 선이 여기에 형성됩니다.
      cv::fitLine(left_pts, left_line, CV_DIST_L2, 0, 0.01, 0.01); //0.01 0.01
      left_m = left_line[1] / left_line[0];
      left_b = cv::Point(left_line[2], left_line[3]);
    }
  }
  
  // 기울기와 옵셋 점 중 하나를 얻은 다음 선 방정식을 적용하여 선 점을 얻습니다
  int ini_y = original.rows;
  int fin_y = 530; //470

  double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
  double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;

  double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
  double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;

  double center_ini_x = (right_ini_x + left_ini_x) / 2;
  double center_fin_x = (right_fin_x + left_fin_x) / 2;

  output[0] = cv::Point(right_ini_x, ini_y);
  output[1] = cv::Point(right_fin_x, fin_y);
  output[2] = cv::Point(left_ini_x, ini_y);
  output[3] = cv::Point(left_fin_x, fin_y); 
  output[4] = cv::Point(center_ini_x, ini_y); // 밑점
  output[5] = cv::Point(center_fin_x, fin_y); // 윗점
  
  return output;
}

int main()
{
	
	VideoCapture cap(0);
	vector<KeyPoint> keypoints1, keypoints2, keypoints3, keypoints4; 
	
	Mat  sobel, grad_x, grad_y, roi1, result1, masked1, eroded1,canny1;
	Mat original, grayoriginal,roi2,roi3,masked2, eroded2,erodedimg1,erodedimg2, erodedimg3,result2,result3,canny2,gausssian;
	Mat  descriptors1, descriptors2,descriptors3,descriptors4;
	Mat cannyImage1,cannyImage2,cannyImage3,resultImage1,resultImage2,resultImage3,gausssian1,gausssian2,gausssian3,gausssianimg1,gausssianimg2,gausssianimg3;
	
	vector<KeyPoint>* keypoints[4]={&keypoints1, &keypoints2, &keypoints3, &keypoints4};
	Mat * descriptorses[4]={ &descriptors1, &descriptors2,&descriptors3,&descriptors4};
	Mat * signMachings[4]={&cannyImage1,&cannyImage2,&cannyImage3,&roi2};
	
	//vector<KeyPoint>* keypoints[4]={&keypoints1, &keypoints2, &keypoints3, &keypoints4};
	//Mat * descriptorses[4]={ &descriptors1, &descriptors2,&descriptors3,&descriptors4};
	//Mat * signMachings[4]={&cannyImage1,&cannyImage2,&cannyImage3,&roi2};

	Mat moriginal, mGray, mCanny, imageROI,mGray1, mGray2, carTrack , mask3, IPM_ROI, IPM, IPM_Gray, IPM1, IPM2 ,IPM_Gray2, moriginal2, masked3, eroded3;
	CascadeClassifier cars, traffic_light, stop_sign, pedestrian,sign, sign2;
	vector<Rect> cars_found, traffic_light_found, stop_sign_found ,pedestrian_found ,sign_found, sign_found2, cars_tracking;
    vector<Mat> cars_tracking_img;
    vector<int> car_timer;

	Ptr<AKAZE> kazeF = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
/*******************************/
/*************car select********/	
	//cars.load(CASCADE_FILE_NAME1);
	//cars.load(CASCADE_FILE_NAME2);
	cars.load(CASCADE_FILE_NAME3);
	//cars.load(CASCADE_FILE_NAME4);
	//cars.load(CASCADE_FILE_NAME5);
	//cars.load(CASCADE_FILE_NAME6);
/********************************/
/********************************/
   
 
				
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat srcImage1,srcImage2,srcImage3;
	Mat Image1 = imread("2_80_3.jpg" );// 기준영상 
	Mat Image2 = imread("80_2.jpg");// 기준영상 
	Mat Image3 = imread("80_6.jpg" );// 기준영상 
	Mat Image4 = imread("80_3_1.jpg" );
	
	cvtColor(Image1, srcImage1, CV_BGR2GRAY); 
	cvtColor(Image2, srcImage2, CV_BGR2GRAY); 
	cvtColor(Image3, srcImage3, CV_BGR2GRAY); 

	GaussianBlur(srcImage1 ,gausssianimg1, Size(5,5),10);	
	GaussianBlur(srcImage2 ,gausssianimg2, Size(5,5),10);	
	GaussianBlur(srcImage3 ,gausssianimg3, Size(5,5),10);	
	//imshow("gausssianimg1", gausssianimg1);

	threshold(gausssianimg1, resultImage1, 150, 255, THRESH_BINARY | THRESH_OTSU); //오츠의 임계 값
	threshold(gausssianimg2, resultImage2, 150, 255, THRESH_BINARY | THRESH_OTSU); //오츠의 임계 값
	threshold(gausssianimg3, resultImage3, 150, 255, THRESH_BINARY | THRESH_OTSU); //오츠의 임계 값	

	erode(resultImage1, erodedimg1, kernel);
	erode(resultImage2, erodedimg2, kernel);
	erode(resultImage3, erodedimg3, kernel);

	Canny(erodedimg1, cannyImage1, 150, 1000, 5); // 가장자리 감지
	Canny(erodedimg2, cannyImage2, 150, 1000, 5);	
	Canny(erodedimg3, cannyImage3, 150, 1000, 5);
	//imshow("canny1", cannyImage1);
	//imshow("canny2", cannyImage2);
	//imshow("canny3", cannyImage3);
			
	kazeF->detectAndCompute(cannyImage1,noArray(),keypoints1, descriptors1);
	kazeF->detectAndCompute(cannyImage2,noArray(),keypoints2, descriptors2);
	kazeF->detectAndCompute(cannyImage3,noArray(),keypoints3, descriptors3);				


	int fps = (int)(cap.get(CAP_PROP_FPS));
	float rho, theta, theta_sum, theta_sum2, theta_mean, theta_mean2, aR,aL, bR,bL, xR0, yR0, xL0,yL0, mR,mL,pR1x,pR1y,pR2x,pR2y, pL1x,pL1y, pL2x,pL2y ,crossX,crossY, cL, cR;
	float rho_sum, rho_sum2, rho_mean, rho_mean2;
	int cnt = 0, cnt2 = 0;
	theta_sum = 0, theta_sum2 = 0;
	rho_sum = 0, rho_sum2 = 0;
	int delay = 1000 / fps;
	Point pL1, pL2, pR1, pR2,crossPoint;

	//vector<Vec2f> lines, lines2;
/********************************/
 /*************비디오 선택**********/
	//cap >> original;//웹캠
	//cap.open(VIDEO_FILE_NAME);//동영상
	//cap.open(VIDEO_FILE_NAME1);thread
	//cap.open(VIDEO_FILE_NAME2);//캡쳐
	//cap.open(VIDEO_FILE_NAME3);//캡쳐
	//cap.open(VIDEO_FILE_NAME4);
	//cap.open(VIDEO_FILE_NAME5);
	//cap.open(VIDEO_FILE_NAME6);
	//cap.open(VIDEO_FILE_NAME7);
	//cap.open(VIDEO_FILE_NAME8);
	cap.open(VIDEO_FILE_NAME9);
/*******************************/
/*******************************/
	//imshow("hi", mask1);
 

	LaneDetector lanedetector;  // 클래스 객체 만들기
 	cv::Mat img_denoise;
 	cv::Mat img_edges;
 	cv::Mat img_mask;
 	cv::Mat img_lines;
 	std::vector<cv::Vec4i> lines;
    	std::vector<std::vector<cv::Vec4i> > left_right_lines;
 
    	std::vector<cv::Point> lane;
    	std::string turn;
    	int flag_plot = -1;
    	int i = 0;
	thread_start(6);
	//system("canberra-gtk-play -f left.ogg"); 
	while (1)
	{	

	       vector< DMatch > goodMatches1,goodMatches2,goodMatches3;
		
		if (!cap.read(original))
		{
			cout << "Input video error!" << endl;
			break;
		}
		if (original.empty())
		{
			cout << "Input original is empty!" << endl;
			break;
		}
		/*************************/
		/*********영상변환 **********/
		
		cvtColor(original, grayoriginal, CV_BGR2GRAY); // 흑백 효과
		GaussianBlur(grayoriginal ,gausssian1, Size(5,5),3);	
		GaussianBlur(grayoriginal ,gausssian2, Size(1,1),5);	
		GaussianBlur(grayoriginal ,gausssian3, Size(5,5),7);
		threshold(gausssian1, result1, 150, 255, THRESH_BINARY | THRESH_OTSU); //오츠의 임계 값
		threshold(grayoriginal, result2, 150,400, THRESH_BINARY | THRESH_OTSU); //오츠의 임계 값
		
		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));


/*********표지판 영상 처리******(***********/
		
		erode(result2, eroded2, kernel); //'커널'- 구조화 요소로 이미지를 옮깁니다.
		Canny(eroded2,canny2, 5000, 500, 5); // 가장자리 감지	
		Rect rect2((original.size().width / 3)*2,original.size().height/4 ,original.size().width /3,original.size().height/3 );// 오른 쪽화면만 분할  
		////cout << "size = " << (original.size().width )<< "x" << (original.size().height) << std:: endl;
		roi2=canny2(rect2); //Set ROI
		
		Mat mask2 = Mat::zeros(original.size(), original.type()); //Mask to isolate ROI
 		cvtColor(mask2, mask2, CV_BGR2GRAY);
		rectangle(mask2, rect2, 255, CV_FILLED, 8); //Drawing a rectangular mask 
		bitwise_and(canny2, mask2, masked2);
/****************************************/
		
/*************차간거리 영상처리 **********/
		threshold(grayoriginal, result3, 100,600, THRESH_BINARY | THRESH_OTSU); //오츠의 임계 값 	
		Rect rect3(original.size().width/5,original.size().height/3,(original.size().width/5)*3,original.size().height/3);
		roi3=gausssian3 (rect3);

		//imshow("cargray",grayoriginal);
		//imshow("carroi",roi3);
		
		Mat mask3 = Mat::zeros(original.size(), original.type()); //Mask to isolate ROI
 		cvtColor(mask3, mask3, CV_BGR2GRAY);
		rectangle(mask3, rect3, 255, CV_FILLED, 8); //Drawing a rectangular mask 
		bitwise_and( grayoriginal, mask3, masked3);
		
/*************************************/		
		/*********테스트 출력부분 *********/	
		//imshow("roi", roi1);
		//int w= 3 * roi.size().width;
		imshow("canny", roi3);
		imshow("masked", mask3);
		imshow("grayoriginal", grayoriginal);
		imshow("gausssian",gausssian3);
		//imshow("carroi",roi);
		/******************************/
		/***********검출 알고리즘 시작******/
		
/***********차간거리검출 *********/
		
		cars.detectMultiScale(roi3, cars_found, 1.1, 7, 0, Size(30, 30));
		draw_locations(original, cars_found, Scalar(0, 255, 0),"Car");
/*********************************/			
/**************carLIne*****************/


	img_denoise = lanedetector.deNoise(original);// 가우스 필터를 사용하여 이미지를 노이즈 제거합니다.

	img_edges = lanedetector.edgeDetector(img_denoise);// 이미지의 가장자리 감지

	img_mask = lanedetector.mask(img_edges);// ROI 만 얻을 수 있도록 이미지를가립니다.
	cv::namedWindow("Lane", CV_WINDOW_AUTOSIZE);
	cv::imshow("mask", img_mask);
	lines = lanedetector.houghLines(img_mask);// 자른 이미지에서 Hough 선을 얻습니다.


        left_right_lines = lanedetector.lineSeparation(lines, img_edges);// 선을 왼쪽과 오른쪽 선으로 구분하십시오.

        lane = lanedetector.regression(left_right_lines, original);// 차선의 각면에 대해 한 줄만 얻으려면 회귀를 적용하십시오.
	
        flag_plot = lanedetector.plotLane(original, lane, turn);// 플롯 레인 탐지
/*******************carLIne end********************/

/*******************표지판 검출 알고리즘*****************/
		
		
			kazeF->detectAndCompute(roi2,noArray(),keypoints4, descriptors4);
			
			goodMatches1 = matching (goodMatches1,descriptors1, descriptors4);/////////매칭
			goodMatches2 = matching (goodMatches2,descriptors2, descriptors4);
			goodMatches3 = matching (goodMatches3,descriptors3, descriptors4);

		if(goodMatches1.size()> 4)
			 {
				vector<Point2f> sceneP1(4);
				sceneP1 =betweenPointAndPoint(srcImage1, keypoints1, keypoints4, goodMatches1,sceneP1);
				makeSqare (original,roi2, sceneP1); // 네모 만들기
				cout << "\t\tsamesame=no.1Image \n\n" <<endl;
				thread_start(5);
			}

		if(goodMatches2.size()> 4)
			 {
				vector<Point2f> sceneP2(4);
				sceneP2 =betweenPointAndPoint(srcImage2, keypoints2, keypoints4, goodMatches2,sceneP2);
				makeSqare (original,roi2, sceneP2); // 네모 만들기
				cout << "\t\tsamesame=no.2Image \n\n" ;
				thread_start(5);
			}
		if(goodMatches3.size()> 4)
			 {
				vector<Point2f> sceneP3(4);
				sceneP3 =betweenPointAndPoint(srcImage3, keypoints3, keypoints4, goodMatches3,sceneP3);
				makeSqare (original,roi2, sceneP3); // 네모 만들기
				cout << "\t\tsamesame=no.3Image \n\n" ;
				thread_start(5);
			}	
	
	
		//imshow("CANNY_EDGE", canny1);
		imshow("ORIGINAL", original);
		waitKey(delay/10);
	}
}
/**************************************************************************/
/**************************메인문 종료  ,함수 들*****************************/
/**************************************************************************/

void signMatching(vector<KeyPoint> *keypoints[4], Mat *descriptorses[4], Mat *signMachings[4])
		{
			Ptr<AKAZE> kazeF = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
			//for(int i=0; i<4; i++)
			   // kazeF->detectAndCompute(*signMachings[i],noArray(),&keypoints[i], &descriptorses[i]);
		}	  


vector< DMatch > matching (vector< DMatch > goodMatches,Mat descriptors1,Mat descriptors2)/////////매칭
{ 
	int k=2;
	vector< vector< DMatch > > matches;
 
	Ptr<DescriptorMatcher> matcher;
	matcher = DescriptorMatcher::create("BruteForce");
 
	matcher->knnMatch( descriptors1, descriptors2, matches, k);
	//cout << "matches.size()=" <<  matches.size() << endl;	

	float nndrRatio = 0.6f;
 	for(unsigned int i = 0; i < matches.size(); i++ )
	{
	//cout << "matches[i].size()=" << matches[i].size() << endl;
		if(matches.at(i).size() == 2 &&
		   matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
		{
			goodMatches.push_back(matches[i][0]); 
		}
	}
	//cout << "goodMatches.size()=" <<  goodMatches.size() << endl;
	return goodMatches;
}


vector<Point2f> betweenPointAndPoint(Mat mat,vector<KeyPoint> keypoints1,vector<KeyPoint> keypoints2, vector< DMatch > goodMatches,vector<Point2f> sceneP){

	vector<Point2f> obj;
	vector<Point2f> scene;
	for( unsigned int i = 0; i < goodMatches.size(); i++ )
	{
	    // Get the keypoints from the good matches
		obj.push_back( keypoints1[ goodMatches[i].queryIdx ].pt );
		scene.push_back( keypoints2[ goodMatches[i].trainIdx ].pt );
	}
	Mat H = findHomography( obj, scene, RANSAC ); //CV_RANSAC
 
	vector<Point2f> objP(4);

	objP[0] = Point2f(0,0);
	objP[1] = Point2f( mat.cols , 0);
	objP[2] = Point2f( mat.cols , mat.rows ); 
	objP[3] = Point2f( 0, mat.rows );
 
	
	perspectiveTransform(objP, sceneP, H);
	return sceneP;
 	}

void makeSqare (Mat mat,Mat roi,vector<Point2f> sceneP) // 네모 만들기
	{ 
	for( int i = 0; i < 4; i++ )	
		sceneP[i]=sceneP[i]+Point2f((roi.size().width * 2),mat.size().height/4);
	
	for( int i = 0; i < 4; i++ )
	 line(mat, sceneP[i], sceneP[(i+1)%4], Scalar(255,0, 0), 4);
		
	
	}



/** kazeF **/
 void kkazeF(Ptr<AKAZE> kazeF, Mat srcImage1,Mat roi,vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,Mat descriptors1,Mat descriptors2){

		kazeF->detectAndCompute(srcImage1,noArray(), keypoints1, descriptors1);
		
		kazeF->detectAndCompute(roi,noArray(), keypoints2, descriptors2);	
		 
}


Point GetWrappedPoint(Mat M, const Point& p)
{
 cv::Mat_<double> src(3/*rows*/,1 /* cols */);
    
 src(0,0)=p.x;
 src(1,0)=p.y;
 src(2,0)=1.0;
   
 cv::Mat_<double> dst = M*src;
 dst(0,0) /= dst(2,0);
 dst(1,0) /= dst(2,0);
 return Point(dst(0,0),dst(1,0));
}

void draw_locations(Mat & img, vector< Rect > &locations, const Scalar & color, string text)
{

    Mat img1, car, carMask ,carMaskInv,car1,roi1, LeftArrow , LeftMask, RightArrow,RightMask;
   
    img.copyTo(img1);
    string dis;
	if (!locations.empty())
	{

        double distance= 0;
        
        for( int i = 0 ; i < locations.size() ; i++){
            
            if (text=="Car"){
                car = imread(CAR_IMAGE);
                carMask = car.clone();
                cvtColor(carMask, carMask, CV_BGR2GRAY);
                locations[i].y = locations[i].y + img1.rows/3; // shift the bounding box
                locations[i].x = locations[i].x + img1.cols/5; // shift the bounding box
                distance = (0.0397*2)/((locations[i].width)*0.00007);// 2 is avg. width of the car                
                Size size(locations[i].width/1.5, locations[i].height/3);
                resize(car,car,size, INTER_NEAREST);
                resize(carMask,carMask,size, INTER_NEAREST);
               
		Mat roi = img.rowRange(locations[i].y-size.height, (locations[i].y+locations[i].height/3)-size.height).colRange(locations[i].x, (locations[i].x  +locations[i].width/1.5));
                //bitwise_and(car, roi, car);
                if(distance < 10)
                {
                    car.setTo(Scalar(0, 0, 255), carMask);
		
                }
                else
                {
                    car.setTo(color, carMask);
                }
 
                add(roi,car,car);
                car.copyTo(img1.rowRange(locations[i].y-size.height, (locations[i].y+locations[i].height/3)-size.height).colRange(locations[i].x, (locations[i].x  +locations[i].width/1.5)));
               // imshow("Test",roi);
                
                }
            stringstream stream;
            stream << fixed << setprecision(2) << distance;
            dis = stream.str() + "m";
            if(distance > 10) {  
               rectangle(img,locations[i], color, -1);
           } else {
               rectangle(img,locations[i], Scalar(0, 0, 255), -1);
            }
        }
        addWeighted(img1, 0.8, img, 0.2, 0, img);
        
        for( int i = 0 ; i < locations.size() ; ++i){
            if(distance > 10) {
                rectangle(img,locations[i],color,1.8);
            } else {
                rectangle(img,locations[i],Scalar(0, 0, 255),1.8);
/**thread**/               
	
      			     }
            putText(img, text, Point(locations[i].x+1,locations[i].y+8), FONT_HERSHEY_DUPLEX, 0.3, color, 1);
            putText(img, dis, Point(locations[i].x,locations[i].y+locations[i].height-5), FONT_HERSHEY_DUPLEX, 0.3, Scalar(0, 255, 255), 1);
            
            
            if (text=="Car"){
                locations[i].y = locations[i].y - img.rows/2; // shift the bounding box
            }
        
        }
	}


}
/**********************************************************/
/**********************KIU*********************************/



void thread_start(int value)
{	
	cout << value << endl;
	switch(value)
	{
		case 1:	
			if(lineRFlag==1)
			{
				lineRFlag = 0;
				thread* lineL =  new thread(thread_function, value);    				
				 cout<< "thread1"<<endl;
				            
 			}	
			break;
		
		case 2:
			if(lineLFlag==1)
			{
				lineLFlag = 0;
		//int w= 3 * roi.size().width;
				
				thread* lineR =  new thread(thread_function, value);
				cout<< "thread2"<<endl;
			}			
			break;
		case 3: 
			if (LineCFlag ==1)
			{
				sign60Flag = 0;
				thread* lineC =  new thread(thread_function, value);
				cout<< "thread3"<<endl;
			}			
			break;
		case 4:
			if(sign60Flag==1)
			{
				sign80Flag = 0;
				thread* sign60 =  new thread(thread_function, value);
				cout<< "thread4"<<endl;
			}
			break;
		case 5:
			if(sign70Flag==1)
			{
				sign70Flag = 0;
				thread* sign70=  new thread(thread_function, value);
				cout<< "thread5"<<endl;
			}	
			break;
		case 6:
			if(sign80Flag==1)
			{
				sign80Flag = 0;
				thread* sign80 =  new thread(thread_function, value);
				cout<< "thread6"<<endl;
				
			}
			break;
				
		default:
			break;
 	}
    

}


void thread_function(int value)
{
	
 switch(value)
	{
		case 1:
			if(lineRFlag==0)// 
			{	
			system("canberra-gtk-play -f left.ogg"); 	
 			lineRFlag = 1;
			}
			break;
		
		case 2:
			if(lineLFlag==0)
			{
			system("canberra-gtk-play -f right.ogg"); 		
			lineLFlag = 1;
			}
			break;
		case 3:
			if(LineCFlag==0);
			{
				system("canberra-gtk-play -f speed60.ogg"); 
 				LineCFlag= 1;			
			}			
			break;
		case 4:
			if(sign60Flag==0)
			{
				system("canberra-gtk-play -f speed70.ogg");  
 				sign60Flag = 1;
			}
			break;
		case 5:	
			if(sign70Flag==0)
			{
				system("canberra-gtk-play -f speed80.ogg"); 
 				sign70Flag = 1;
			}
			break;
		case 6:if(sign80Flag==0)// what`s up ma
			{			
				system("canberra-gtk-play -f FirstNav.ogg"); 
 				sign80Flag = 1;
			}	
			break;
		
		default:
			break;		
 	}
}






/***********************************************************************/
/***********************************KI_U********************************/
/***********************************************************************/

int LaneDetector::plotLane(cv::Mat original, std::vector<cv::Point> lane, std::string turn) {
  std::vector<cv::Point> poly_points;

  cv::Mat output;

  double vanish_x; // 소실점 저장하는 변수
  double thr_vp = 50; // 소실점에서 +, - 할 값
  
  vanish_x = static_cast<double>(((right_m*right_b.x) - (left_m*left_b.x) - right_b.y + left_b.y) / (right_m - left_m));// 소실점은 두 차선 경계선이 교차하는 점입니다.

  // 레인의 더 나은 시각화를 위해 투명한 다각형을 만듭니다.
  original.copyTo(output);
  poly_points.push_back(lane[2]);
  poly_points.push_back(lane[0]);
  poly_points.push_back(lane[1]);
  poly_points.push_back(lane[3]);
 
  // 소실점 위치에 따라 선회하는 도로가 결정됩니다.
  if (vanish_x < (img_center - thr_vp)){
    cv::fillConvexPoly(output, poly_points, cv::Scalar(0, 0, 255), CV_AA, 0); //poly_points좌표 안에 색 채우기
    cv::addWeighted(output, 0.3, original, 1.0 - 0.3, 0, original); // fillConvexPoly에 채워진 색 투명하게
    cv::putText(original, "Right Warning", cv::Point(510, 150), cv::FONT_HERSHEY_DUPLEX, 2, cvScalar(0, 0, 255), 3, CV_AA); // 텍스트 출력
	thread_start(1);
  }
  else if (vanish_x > (img_center + thr_vp)){
    cv::fillConvexPoly(output, poly_points, cv::Scalar(0, 0, 255), CV_AA, 0); //poly_points좌표 안에 색 채우기
    cv::addWeighted(output, 0.3, original, 1.0 - 0.3, 0, original); // fillConvexPoly에 채워진 색 투명하게
    cv::putText(original, "Left Warning", cv::Point(510, 150), cv::FONT_HERSHEY_DUPLEX, 2, cvScalar(0, 0, 255), 3, CV_AA); // 텍스트 출력
	thread_start(2);
  }
  else if (vanish_x >= (img_center - thr_vp) && vanish_x <= (img_center + thr_vp)){ 
    cv::fillConvexPoly(output, poly_points, cv::Scalar(0, 255, 0), CV_AA, 0); //poly_points좌표 안에 색 채우기
    cv::addWeighted(output, 0.3, original, 1.0 - 0.3, 0, original); // fillConvexPoly에 채워진 색 투명하게
    cv::putText(original, "Safety", cv::Point(510, 150), cv::FONT_HERSHEY_DUPLEX, 2, cvScalar(0, 255, 0), 3, CV_AA); //텍스트 출력
  }
  
  // 차선 경계의 두 선을 그려라.
  cv::line(original, lane[0], lane[1], cv::Scalar(0, 255, 255), 5, CV_AA);   //왼쪽 차선 그리기 
  cv::line(original, lane[2], lane[3], cv::Scalar(0, 255, 255), 5, CV_AA);   //오른쪽 차선 그리기
  //cv::line(original, lane[4], lane[5], cv::Scalar(255, 255, 255), 5, CV_AA);   // 왼쪽, 오른쪽 차선의 중심점 그리기

  //최종 출력 이미지 표시
  //cv::namedWindow("Lane", CV_WINDOW_AUTOSIZE);
  //cv::imshow("Lane", original);


  return 0;
}

