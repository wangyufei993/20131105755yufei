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

//ʵ��һ����ʾͼƬ

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

//ʵ�������ʾͼƬ����
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
//ʵ������ֱ�߼��
/* void drawDetectLines(Mat& image,const vector<Vec4i>& lines,Scalar & color)  
{  
// ����⵽��ֱ����ͼ�ϻ�����  
vector<Vec4i>::const_iterator it=lines.begin();  
while(it!=lines.end())  
{  
Point pt1((*it)[0],(*it)[1]);  
Point pt2((*it)[2],(*it)[3]);  
line(image,pt1,pt2,color,2); //  �����������Ϊ2  
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
// ���ֱ�ߣ���СͶƱΪ90������������50����϶��С��10  
HoughLinesP(contours,lines,1,CV_PI/180,80,50,10);  
drawDetectLines(image,lines,Scalar(0,255,0));  

namedWindow("zhixian");  
imshow("zhixian",image);  
waitKey();  
return 0;  
}  */



//ʵ���ģ��Ƚ�����ͼƬ�����ƶȣ��Ҷ�ͼ��
int HistogramBins = 256;  //ֱ��ͼά������Ŀ,Ҳ�Ͱ�����ͼ�ֳ���256������
float HistogramRange1[2]={0,255};  //�Ҷ�ֵ(�趨����ֵ��Χ) Ϊ0-255
float *HistogramRange[1]={&HistogramRange1[0]};  //��һ����ʶ����ԭ���е㸴�ӡ�ͨ��ʹ��Ĭ��ֵ����
int CompareHist()  
{  
	int size = 256; //ֱ��ͼ�ߴ� 
	int scale = 2;//���ظ��������������ţ����ű���Ϊ2
	int height = 256;
	IplImage * image1=cvLoadImage("E:\\1.jpg",0);  //0:��ʾ��ͨ��ͼ��
	IplImage * image2=cvLoadImage("E:\\2.jpg",0); 

	//������Ҫһ��ָ��ṹCvHistogram ��ָ��Histogram�����ָ����ʹ��֮ǰ���뾭����ʼ�����������Ƿ�����ռ��
	CvHistogram *Histogram1 = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
	CvHistogram *Histogram2 = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
	//CV_HIST_ARRAY����ʾֱ��ͼ�����Ƕ�ά�ܼ�����

	cvCalcHist(&image1, Histogram1); //cvCalcHist����ͼ�����Զ�����ֱ��ͼ��imageΪͳ��ͼ��histΪֱ��ͼ�ṹ��
	cvCalcHist(&image2, Histogram2);  

	cvNormalizeHist(Histogram1, 1); //��һ��ֱ��ͼ ,1��ʾ�Ҷ�ͼ
	cvNormalizeHist(Histogram2, 1);  

	IplImage* hist_img1= cvCreateImage(cvSize(size* scale, height), 8, 1);//������ͨ����8λͼ��cvSize���ο��С��������Ϊ����
	IplImage* hist_img2= cvCreateImage(cvSize(size* scale, height), 8, 1);
	//����һ��һάֱ��ͼ�ġ�ͼ����������Ϊ�Ҷȼ���������Ϊ���ظ�����*scale��

	cvZero(hist_img1);//�൱�ڳ�ʼ��ͼƬ���������������Ϊ0, ��Ȼ�ܿ����������
	cvZero(hist_img2);

	float max_value= 0;//����ֱ��ͼ����󷽿�ֵ,��ʼ��Ϊ0��max_value��ֱ��ͼ���ֵ��ָ��

	//�����濪ʼ����ֱ��ͼ��

	cvGetMinMaxHistValue(Histogram1, 0, &max_value, 0, 0);//ֻ�����ֵ

	//�ֱ�ÿ��ֱ�����ֵ���Ƶ�ֱ��ͼͼ��
	for(int i=0; i<size; i++)
	{
		float bin_val= cvQueryHistValue_1D(Histogram1,i);   //����i�ĸ���    ������Ϊi��ֱ�����С   
		int intensity = cvRound(bin_val* height/ max_value);  // Ҫ���Ƶĸ߶�
		//����i�Ҷȼ�������  
		cvRectangle(hist_img1,//Ҫ���Ƶ�����ͼ��
			cvPoint(i* scale, height- 1),//���ε�һ�����㣬���½�
			cvPoint((i+1)* scale- 1, height- intensity),//���ζԽ����ϵ���һ�����㣬���Ͻ�
			CV_RGB(255, 255, 255));//������ɫ
	}
	cvGetMinMaxHistValue(Histogram2, 0, &max_value, 0, 0);
	//����ֱ��ͼ
	for(int i=0; i<size; i++)
	{
		float bin_val= cvQueryHistValue_1D(Histogram1,i);   //����i�ĸ���
		int intensity = cvRound(bin_val* height/ max_value);  // ���Ƶĸ߶�
		cvRectangle(hist_img2,
			cvPoint(i* scale, height- 1),
			cvPoint((i+1)* scale- 1, height- intensity),
			CV_RGB(255, 255, 255));
	}

	// CV_COMP_CHISQR,CV_COMP_BHATTACHARYYA�����ֶ�����������ֱ��ͼ�ıȽϣ�ֵԽС��˵��ͼ��Խ����  
	printf("CV_COMP_CHISQR : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_CHISQR));  
	printf("CV_COMP_BHATTACHARYYA : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_BHATTACHARYYA));  


	// CV_COMP_CORREL, CV_COMP_INTERSECT������ֱ��ͼ�ıȽϣ�ֵԽ��˵��ͼ��Խ����  
	printf("CV_COMP_CORREL : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_CORREL));  
	printf("CV_COMP_INTERSECT : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_INTERSECT));  

	cvShowImage("tupian1",image1);
	cvShowImage("tupian2",image2);
	cvShowImage("zhifangtu1",hist_img1);
	cvShowImage("zhifangtu2",hist_img2);
	cvWaitKey();//�ȴ�������cvWaitKey()�����Ĺ������ǳ�����ͣ���ȴ��û�����һ������������
	cvReleaseImage(&image1);  //�����Ѷ����IplImageָ��������ͷ�ռ���ڴ�ռ�
	cvReleaseImage(&image2);  
	cvReleaseHist(&Histogram1);  
	cvReleaseHist(&Histogram2);  
	return 0;  
}  
int main(int argc, char** argv)  //������������в���,�������������ڽ��ܲ����ͼ�¼������Ϣ��
{  
	CompareHist();  
	return 0;  
}