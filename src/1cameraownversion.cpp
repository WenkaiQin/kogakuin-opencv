// 1cameraownversion.cpp

// Import basic C++ libraries.
#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include <math.h>

// Import OpenCV specific libraries.
// #include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Use both std and cv namespaces.
using namespace std;
using namespace cv;

// 抽出したい色の指定
Mat g_min(1, 1, CV_8UC3, Scalar(30,70,30));
Mat g_max(1, 1, CV_8UC3, Scalar(70,255,255));

// 膨張化用のカーネル
Mat k = Mat::ones(5, 5, CV_8UC1);

bool DEBUG = false;


Mat colorTrack(Mat& im, Mat& h_min, Mat& h_max) {
	Mat im_h, mask, im_c;
	int iterations = 2;
	cvtColor(im, im_h, COLOR_BGR2HSV);		// RGB色空間からHSV色空間に変換
	inRange(im_h, h_min, h_max, mask);		// マスク画像の生成
	// medianBlur(mask, mask, 7);			// 平滑化
	// dilate(mask, mask, k);				// 膨張化
	// bitwise_and(im, im, im_c, mask);		// 色領域抽出
	return mask;
}


int indexEmax(vector<vector<Point> >& cnt) {

	vector<vector<Point> >::size_type max_num = 0;
	vector<vector<Point> >::size_type cnt_num = 0;
	int max_i = -1;

	for(int i = 0; i < cnt.size(); i++) {
		cnt_num = cnt[i].size();
		if(cnt_num > max_num) {
			max_num = cnt_num;
			max_i = i;
		}
	}

	return max_i;
}


array<vector<int>,2> where255(Mat& mask) {

	array<vector<int>,2> whereCoords;

	CV_Assert(mask.depth()==CV_8U && mask.channels()==1);
	int nRows = mask.rows, nCols = mask.cols;

	uchar* rowp;
	for(int i = 0; i < nRows; i++) {
		rowp = mask.ptr<uchar>(i);
		for(int j = 0; j < nCols; j++) {
			if(rowp[j]==255) {
				whereCoords[0].push_back(i);
				whereCoords[1].push_back(j);
			}
		}
	}

	return whereCoords;
}


int main() {

	Mat im, mask1, mask2, mask;
	vector<vector<Point> > cnt, hull;

	int h1 = 175;
	int L1 = 700;
	int H = 167;							// 緑色のボトルの実際の高さ[mm]
	int Z = 0;

	VideoCapture cap(0);

	while(1) {

		// 入力画像の取得
		cap >> im;

		// 紅色のカラートラッキング
		mask1 = colorTrack(im, g_min, g_max);
		mask2 = colorTrack(im, g_min, g_max);

		// 紅色領域の輪郭を抽出
		findContours(mask2, cnt, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
		int n = indexEmax(cnt);
		int cx, cy;

		if(n != -1) {
			vector<vector<Point> > hull(1);
			convexHull(Mat(cnt[n]), hull[0]);
			mask2 = Mat::zeros(mask2.size(), mask2.type());
			drawContours(mask2, hull, 0, Scalar(255), -1);
			drawContours(im, hull, 0, Scalar(0,200,0), 3);
			Moments M1 = moments(cnt[n]);
			cx = int(M1.m10/M1.m00), cy = int(M1.m01/M1.m00);
		} else {
			cx = 320, cy = 240;
		}

		// 紅色領域の高さ計算
		bitwise_and(mask1, mask1, mask, mask2);
		Canny(mask2, mask, 100, 200);

		array<vector<int>,2> yx = where255(mask);
		vector<int>& y = yx[0];
		vector<int>& x = yx[1];

		if(y.size()!=0) {

			// 対象物体の画像上の高さh2を計算
			int ymax = *max_element(y.begin(), y.end());
			int ymin = *min_element(y.begin(), y.end());
			int h2 = ymax - ymin;

			// 奥行きL2を計算
			float L2 = (h1/float(h2))*L1;
			if(DEBUG) {
				cout << "h1: " << h1 << endl;
				cout << "h2: " << h2 << endl;
				cout << "L1: " << L1 << endl;
				cout << "L2: " << L2 << endl;
			}

			// 1px当たりの大きさを計算
			float a = H/float(h2);

			// 三次元位置（X, Y, Z）を計算
			float X = (cx-320)*a;
			float Y = (ymax-cy)*a;
			if(L2 > X)
				float Z = sqrt(L2*L2 - X*X);
			X = round(X), Y = round(Y), Z = round(Z), L2 = round(L2);

			if(DEBUG) {
				cout << "X: " << X << endl;
				cout << "Y: " << Y << endl;
				cout << "Z: " << Z << endl;
				cout << "L2: " << L2 << endl;	
			}

			// 結果表示
			circle(im, Point(cx,cy), 5, Scalar(0,0,255), -1);
            putText(im, "X: " + to_string(X) + "[mm]", Point(30,20), 1, 1.5, Scalar(70,70,220), 2);
            putText(im, "Y: " + to_string(Y) + "[mm]", Point(30,50), 1, 1.5, Scalar(70,70,220), 2);
            putText(im, "Z: " + to_string(Z) + "[mm]", Point(30,80), 1, 1.5, Scalar(70,70,220), 2);
            putText(im, "h2: " + to_string(h2) + "[pixcel]", Point(30,120), 1, 1.5, Scalar(220,70,90), 2);
            putText(im, "L2: " + to_string(L2) + "[mm]", Point(30,160), 1, 1.5, Scalar(220,70,90), 2);
            imshow("Camera", im);
            imshow("Mask", mask2);
		}


		// キーが押されたらループから抜ける
		if(waitKey(10) > 0){
			cap.release();
			destroyAllWindows();
			break;
		}
		
	}

	return 0;
}