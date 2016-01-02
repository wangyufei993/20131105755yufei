#include "stdafx.h"

#include <opencv2\opencv.hpp>  
#include <opencv2/core/core.hpp>   
#include <iostream>  
#include <cv.h>
#include<opencv2/highgui/highgui.hpp>
#include"opencv2/imgproc/imgproc.hpp"
#include <string>  
using namespace cv;
using namespace std;

#define cvQueryHistValue_2D( hist, idx0, idx1 )   cvGetReal2D( (hist)->bins, (idx0), (idx1) )
#define cvQueryHistValue_1D( hist, idx0 ) \
	((float)cvGetReal1D( (hist)->bins, (idx0)))

//实验一：显示图片

/*int main()
{
Mat img = imread("E:\\1.jpg");
if (img.empty())
{
cout << "error";
return -1;
}
imshow("ceshi", img);
waitKey();
return 0;
}*/

//实验二：显示图片轮廓
/*int main()
{
Mat M = imread("E:\\1.jpg");
cvtColor(M,M,CV_BGR2GRAY);
Mat contours;
Canny(M,contours,125,350);
threshold(contours,contours,128,155,THRESH_BINARY);
namedWindow("lunkuo");
imshow("lunkuo", contours);
waitKey();
return 0;
return 0;
}*/
//实验三，直线检测
/* void drawDetectLines(Mat& image,const vector<Vec4i>& lines,Scalar & color)  
{  
// 将检测到的直线在图上画出来  
vector<Vec4i>::const_iterator it=lines.begin();  
while(it!=lines.end())  
{  
Point pt1((*it)[0],(*it)[1]);  
Point pt2((*it)[2],(*it)[3]);  
line(image,pt1,pt2,color,2); //  线条宽度设置为2  
++it;  
}  
} 
int main()
{  
Mat image=imread("E:\\3.jpg");  
Mat I;  
cvtColor(image,I,CV_BGR2GRAY);  

Mat contours;  
Canny(I,contours,125,350);  
threshold(contours,contours,128,255,THRESH_BINARY);  

vector<Vec4i> lines;  
// 检测直线，最小投票为90，线条不短于50，间隙不小于10  
HoughLinesP(contours,lines,1,CV_PI/180,80,50,10);  
drawDetectLines(image,lines,Scalar(0,255,0));  

namedWindow("zhixian");  
imshow("zhixian",image);  
waitKey();  
return 0;  
}  */



//实验四：比较两张图片的相似度（灰度图）
int HistogramBins = 256;  //直方图维数的数目,也就把这张图分成了256个区域
float HistogramRange1[2]={0,255};  //灰度值(设定像素值范围) 为0-255
float *HistogramRange[1]={&HistogramRange1[0]};  //归一化标识，其原理有点复杂。通常使用默认值即可
int CompareHist()  
{  
	int size = 256; //直方图尺寸 
	int scale = 2;//像素个数；按比例缩放，缩放倍数为2
	int height = 256;
	IplImage * image1=cvLoadImage("E:\\1.jpg",0);  //0:表示单通道图像
	IplImage * image2=cvLoadImage("E:\\2.jpg",0); 

	//函数需要一个指向结构CvHistogram 的指针Histogram，这个指针在使用之前必须经过初始化，即必须是分配过空间的
	CvHistogram *Histogram1 = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
	CvHistogram *Histogram2 = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
	//CV_HIST_ARRAY：表示直方图类型是多维密集数组

	cvCalcHist(&image1, Histogram1); //cvCalcHist：从图像中自动计算直方图；image为统计图像，hist为直方图结构体
	cvCalcHist(&image2, Histogram2);  

	cvNormalizeHist(Histogram1, 1); //归一化直方图 ,1表示灰度图
	cvNormalizeHist(Histogram2, 1);  

	IplImage* hist_img1= cvCreateImage(cvSize(size* scale, height), 8, 1);//创建单通道的8位图像；cvSize矩形框大小，以像素为精度
	IplImage* hist_img2= cvCreateImage(cvSize(size* scale, height), 8, 1);
	//创建一张一维直方图的“图”，横坐标为灰度级，纵坐标为像素个数（*scale）

	cvZero(hist_img1);//相当于初始化图片，将数组的内容清为0, 不然很可能是随机数
	cvZero(hist_img2);

	float max_value= 0;//计算直方图的最大方块值,初始化为0；max_value：直方图最大值的指针

	//从下面开始绘制直方图：

	cvGetMinMaxHistValue(Histogram1, 0, &max_value, 0, 0);//只找最大值

	//分别将每个直方块的值绘制到直方图图中
	for(int i=0; i<size; i++)
	{
		float bin_val= cvQueryHistValue_1D(Histogram1,i);   //像素i的概率    ；像素为i的直方块大小   
		int intensity = cvRound(bin_val* height/ max_value);  // 要绘制的高度
		//填充第i灰度级的数据  
		cvRectangle(hist_img1,//要绘制到哪张图上
			cvPoint(i* scale, height- 1),//矩形的一个顶点，左下角
			cvPoint((i+1)* scale- 1, height- intensity),//矩形对角线上的另一个顶点，右上角
			CV_RGB(255, 255, 255));//线条颜色
	}
	cvGetMinMaxHistValue(Histogram2, 0, &max_value, 0, 0);
	//绘制直方图
	for(int i=0; i<size; i++)
	{
		float bin_val= cvQueryHistValue_1D(Histogram1,i);   //像素i的概率
		int intensity = cvRound(bin_val* height/ max_value);  // 绘制的高度
		cvRectangle(hist_img2,
			cvPoint(i* scale, height- 1),
			cvPoint((i+1)* scale- 1, height- intensity),
			CV_RGB(255, 255, 255));
	}

	// CV_COMP_CHISQR,CV_COMP_BHATTACHARYYA这两种都可以用来做直方图的比较，值越小，说明图形越相似  
	printf("CV_COMP_CHISQR : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_CHISQR));  
	printf("CV_COMP_BHATTACHARYYA : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_BHATTACHARYYA));  


	// CV_COMP_CORREL, CV_COMP_INTERSECT这两种直方图的比较，值越大，说明图形越相似  
	printf("CV_COMP_CORREL : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_CORREL));  
	printf("CV_COMP_INTERSECT : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_INTERSECT));  

	cvShowImage("tupian1",image1);
	cvShowImage("tupian2",image2);
	cvShowImage("zhifangtu1",hist_img1);
	cvShowImage("zhifangtu2",hist_img2);
	cvWaitKey();//等待按键：cvWaitKey()函数的功能是是程序暂停，等待用户触发一个按键操作。
	cvReleaseImage(&image1);  //销毁已定义的IplImage指针变量，释放占用内存空间
	cvReleaseImage(&image2);  
	cvReleaseHist(&Histogram1);  
	cvReleaseHist(&Histogram2);  
	return 0;  
}  
int main(int argc, char** argv)  //这个叫做命令行参数,这两个就是用于接受参数和记录参数信息的
{  
	CompareHist();  
	return 0;  
}