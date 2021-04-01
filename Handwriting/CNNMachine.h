#pragma once
#include "framework.h"

class CNNMachine
{
public:
	//훈련 데이터 순서 : 데이터 수, 행, 열
	std::vector<cv::Mat>* trainingMats; //=x1Mats
	std::vector<cv::Mat> x1ZeroPaddingMats;

	//커널 데이터 순서 : 채널 수, 커널 수, 행, 열
	std::vector<std::vector<cv::Mat>> kernels1;
	std::vector<std::vector<cv::Mat>> kernels2;
	cv::Size kernel1Stride;
	cv::Size kernel2Stride;

	//합성곱 데이터 순서 : 데이터 수, 채널 수, 행, 열
	std::vector<std::vector<cv::Mat>> conv1Mats;
	std::vector<std::vector<cv::Mat>> conv1ZeroPaddingMats;
	std::vector<std::vector<cv::Mat>> conv2Mats;
	std::vector<std::vector<cv::Mat>> conv2ZeroPaddingMats;
	
	//풀링결과 데이터 순서 : 데이터 수, 채널 수, 행, 열
	std::vector<std::vector<cv::Mat>> poolresult1;
	std::vector<std::vector<cv::Mat>> poolresultZeroPadding1;
	std::vector<std::vector<cv::Mat>> poolresult2;
	cv::Size poolSize;
	cv::Size poolStride;

	//완전연결신경망 층에서는 행렬로만 계산
	cv::Mat xMat;//=poolresult2를 펼친 형태
	cv::Mat wMat;
	cv::Mat yHatMat;
	cv::Mat yMat;

	double lossAverage;
public:
	void Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec);
	//정방향 계산
	void ForwardPropagation();
	void BackPropagation(cv::InputArray _Input, cv::OutputArray _Output);
	void Training(int epoch, float learningRate, float l2);

};

