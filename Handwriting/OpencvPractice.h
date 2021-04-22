#pragma once

#include "framework.h"
//CV_<bit-depth>{U|S|F}C(number_of_channels)
//U : 부호 없는 정수형, S : 부호 있는 정수형, F : 부동 소수형
//CV_8U		unsigned char (256)
//CV_8S		signed char (-128~127)
//CV_16U	unsigned short (65572)
//CV_16S	signed shor (-32768~32767)
//CV_32S	int
//CV_32F	double
//CV_64F	double
//CV_16F	double16_t (-32768~32767)

//Opencv 테스트 클래스
class OpencvPractice
{
private:
	int fps;
	int delay;
public:
	OpencvPractice() : fps(60), delay(cvRound(1000/fps)) {};

	void ImgSizePrint(cv::String);
	void ImgInfoPrint(cv::String);
	void MatOp1();

	void RandomImage(cv::String);
	void ImagePrint(cv::String);

	void Camera_In();
	void Video_In(cv::String, double = 0.25f);
	void Camera_In_Video_Out(cv::String = "C:\\");

	//---------------------------------
	void FaceScan();
	void MatPrint(std::vector<cv::Mat>&, std::vector<uint8_t>&);
	void MnistTrainingDataRead(std::string, std::vector<cv::Mat>&, int readDataNum = -1);
	void MnistLabelDataRead(std::string, std::vector<uint8_t>&, int readDataNum = -1);
	int ConvertCVGrayImageType(int mNistMagicNumber);

	int ReverseInt(int);

};