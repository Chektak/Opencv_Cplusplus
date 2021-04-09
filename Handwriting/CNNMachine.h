#pragma once
#include "framework.h"

class CNNMachine
{
public:
	//훈련 데이터 순서 : 데이터 수, 행렬
	std::vector<cv::Mat> trainingMats; //=x1Mats
	std::vector<cv::Mat> x1ZeroPaddingMats;//합성곱층 1 입력

	//커널 데이터 순서 : 채널 수, 커널 수, 행렬
	std::vector<std::vector<cv::Mat>> kernels1;//합성곱층 1 입력
	std::vector<std::vector<cv::Mat>> kernels2;//합성곱층 2 입력
	cv::Size kernel1Stride;
	cv::Size kernel2Stride;

	//정방향 합성곱 계산 시 사용하는 행렬들
	//데이터 순서 : 데이터 수, 채널 수, 행렬
	std::vector<std::vector<cv::Mat>> conv1Mats;//합성곱층 1 결과
	std::vector<std::vector<cv::Mat>> conv2Mats;//합성곱층 2 결과
	std::vector<std::vector<cv::Mat>> conv1ZeroPaddingMats;//풀링층 1 입력
	std::vector<std::vector<cv::Mat>> conv2ZeroPaddingMats;//풀링층 2 입력
	
	//정방향 풀링 시 사용하는 행렬들
	//데이터 순서 : 데이터 수, 채널 수, 행렬
	std::vector<std::vector<cv::Mat>> poolresult1;//풀링층 1 결과
	std::vector<std::vector<cv::Mat>> poolresult1ZeroPadding;//합성곱층 2 입력
	std::vector<std::vector<cv::Mat>> poolresult2;//풀링층 2 결과
	cv::Size poolSize;
	cv::Size poolStride;

	//정방향 완전연결신경망 계산 시 사용하는 행렬들
	cv::Mat xMat;//poolresult2를 신경망 입력으로 펼친 형태
	cv::Mat wMat;
	cv::Mat yHatMat;
	cv::Mat yMat;

	//역방향 계산시 사용하는 합성곱 결과 행렬이 Max풀링될 때 필터
	//데이터 순서 : 데이터 수, 커널 수, 행렬
	std::vector<std::vector<cv::Mat>> pool1Filters;
	std::vector<std::vector<cv::Mat>> pool2Filters;

	//역방향 계산시 사용하는 입력 행렬이 커널과 합성곱될 때 필터
	//데이터 순서 : 데이터 수, 채널 수, 행, 열, K에 곱해지는 계수 수
	std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> conv1KernelFilters;
	std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> conv2KernelFilters;

	double lossAverage;
public:
	void Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec);
	//정방향 계산
	void ForwardPropagation();
	void BackPropagation();
	void Training(int epoch, float learningRate, float l2);

};

