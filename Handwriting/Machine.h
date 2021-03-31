#pragma once
#include "framework.h"

class Machine
{
public:
	//커널 데이터 순서 : 커널 수, 행, 열
	std::vector<cv::Mat> kernels1;
	std::vector<cv::Mat> kernels2;

	cv::Size kernel1Stride;
	cv::Size kernel2Stride;

	//합성곱 데이터 순서 : 데이터 수, 행, 열
	std::vector<cv::Mat> conv1;
	std::vector<cv::Mat> conv2;

	cv::Size poolSize;
	cv::Size poolStride;

	//훈련 데이터 순서 : 데이터 수, 행, 열
	std::vector<cv::Mat> x1Mats;
	std::vector<cv::Mat> x2Mats;
	cv::Mat wMat;
	std::vector<cv::Mat> yHatMats;
	std::vector<cv::Mat> yMats;

	double lossAverage;
public:
	void Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec);
	//정방향 계산 후 yHat을 반환
	void ForwardPropagation(cv::InputArray _Input, cv::OutputArray _Output);
	void BackPropagation(cv::InputArray _Input, cv::OutputArray _Output);
	void Training(int epoch, float learningRate, float l2);

};

