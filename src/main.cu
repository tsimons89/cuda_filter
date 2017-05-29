#include <opencv2/opencv.hpp>
#include "gpu_filter.h"
#include <ctime>
#include <stdio.h>
using namespace cv;
#define N 10
int main( void ) {
	Mat test_mat = imread("image.jpg",IMREAD_GRAYSCALE);
	test_mat.convertTo(test_mat,CV_32F);
	float * image,*output;
	output = (float*)malloc(test_mat.total()*sizeof(float));
	image = (float*)test_mat.data;
	clock_t begin = clock();
	gpu_filter(image,output,test_mat.cols,test_mat.rows);
	clock_t end = clock();
	printf("Cycles: %d\n",end-begin);
	begin = clock();
	gpu_filter(image,output,test_mat.cols,test_mat.rows);
	end = clock();
	printf("Cycles: %d\n",end-begin);
	// Mat out_mat(test_mat.size(),CV_32F,output);

	// test_mat.convertTo(test_mat,CV_8U);
	// out_mat.convertTo(out_mat,CV_8U);
	// imshow("in",test_mat);
	// imshow("out",out_mat);
	// waitKey(0);

	begin = clock();
	Sobel(test_mat,test_mat,-1,1,0);
	end = clock();
	printf("Cycles: %d\n",end-begin);

	begin = clock();
	Sobel(test_mat,test_mat,-1,1,0);
	end = clock();
	printf("Cycles: %d\n",end-begin);


	return 0;
}
