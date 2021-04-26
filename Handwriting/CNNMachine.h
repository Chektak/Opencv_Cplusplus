#pragma once
#include "framework.h"

class CNNMachine
{
public:
	//훈련 데이터 순서 : 데이터 수, 행렬
	//(채널 수는 흑백 이미지 입력만을 가정하므로 생략)
	std::vector<cv::Mat> trainingMats; //=x1Mats
	std::vector<cv::Mat> x1ZeroPaddingMats;//합성곱층 1 입력

	//커널 데이터 순서 : 채널 수, 커널 수, 행렬
	std::vector<std::vector<cv::Mat>> kernels1;//합성곱층 1 입력
	std::vector<std::vector<cv::Mat>> kernels2;//합성곱층 2 입력
	cv::Size kernel1Stride;
	cv::Size kernel2Stride;

	//데이터 순서 : 데이터 수, 채널 수, 스칼라
	std::vector<std::vector<double>> conv1Bias;//합성곱층 1 결과에 더할 편향
	std::vector<std::vector<double>> conv2Bias;//합성곱층 2 결과에 더할 편향
	
	//정방향 합성곱 계산 시 사용하는 행렬들
	//데이터 순서 : 데이터 수, 채널 수, 행렬
	std::vector<std::vector<cv::Mat>> conv1Mats;//합성곱층 1 결과
	std::vector<std::vector<cv::Mat>> conv2Mats;//합성곱층 2 결과
	std::vector<std::vector<cv::Mat>> conv1ZeroPaddingMats;//풀링층 1 입력
	std::vector<std::vector<cv::Mat>> conv2ZeroPaddingMats;//풀링층 2 입력
	
	//정방향 풀링 시 사용하는 행렬들
	//데이터 순서 : 데이터 수, 채널 수, 행렬
	std::vector<std::vector<cv::Mat>> pool1result;//풀링층 1 결과
	std::vector<std::vector<cv::Mat>> pool1resultZeroPadding;//합성곱층 2 입력
	std::vector<std::vector<cv::Mat>> pool2result;//풀링층 2 결과
	cv::Size poolSize;
	cv::Size poolStride;
	cv::Size pool1ResultSize;
	cv::Size pool2ResultSize;

	//정방향 완전연결신경망 계산 시 사용하는 행렬들
	cv::Mat xMat;//완전연결신경망 1층 입력 (pool2result를 2차원으로 펼친 형태)
	cv::Mat w1Mat;//완전연결신경망 1층 입력
	double bias1 = 0;
	cv::Mat a1Mat;//완전연결신경망 1층 결과, 완전연결신경망 2층 입력
	cv::Mat w2Mat;//완전연결신경망 2층 입력
	double bias2 = 0;
	cv::Mat yHatMat;//모델 예측값
	cv::Mat yMat;//정답 행렬

#pragma region 역방향 계산에서만 사용
	//역방향 계산시 사용하는 Max풀링 필터(풀링을 입력행렬에 대해 미분)
	//데이터 순서 : 데이터 수, 커널 수, 행렬
	std::vector<std::vector<cv::Mat>> pool1BackpropFilters;
	std::vector<std::vector<cv::Mat>> pool2BackpropFilters;

	/* 합성곱을 커널에 대해 미분할 수 없는 이유								*
	*	: 제로 패딩 행이나 열과 0인 입력행렬을 구분할 수 없음					*
	*	해결책1 : 입력행렬을 제로패딩할 때 원본 입력행렬의 st, ed 포인트를 저장	*
	*	해결책2 : 입력행렬과 제로패딩, 스트라이드, 커널 크기를 분석해			*
	*			수식으로 제로패딩 부분을 알아낸다							*
	*   해결책 1 사용														*/

	//역방향 계산시 사용하는 합성곱 필터(제로 패딩과 입력행렬을 구분하기 위해 좌표를 기록해 사용)
	//데이터 순서 : 합성곱 결과행렬 행*열, pair(커널 기준 x 입력 행렬 start index, 커널 기준 x 입력 행렬 end index)
	std::vector<std::pair<int, int>> conv1BackpropFilters;
	std::vector<std::pair<int, int>> conv2BackpropFilters;

	cv::Mat yLoss; //손실함수를 SoftMax 함수 결과에 대해 미분한 값
	cv::Mat w2T;
	cv::Mat yLossW2;//손실 함수를 완전연결층2 입력(ReLu(aMat))에 대해 미분한 값
	cv::Mat yLossW2Relu3; //손실함수를 완전연결층1 결과에 대해 미분한 값
	cv::Mat yLossW2Relu3W1; //손실 함수를 완전연결층1 입력에 대해 미분한 값

	std::vector<std::vector<cv::Mat>> yLossW2Relu3W1UpRelu2; //손실 함수를 합성곱2 결과에 대해 미분한 값 (Up은 Up-Sampleling(풀링함수의 미분))
	std::vector<std::vector<cv::Mat>> yLossW2Relu3W1UpRelu2P1UpRelu; //손실 함수를 합성곱1 결과에 대해 미분한 값
#pragma endregion
	std::vector<double> lossAverages;  
	double loss;
	double learningRate;
public:
	void Training(int epoch, double learningRate, double l2);
	void Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec);
	//정방향 계산
	void ForwardPropagation();
	void BackPropagation(double learningRate);
	void SaveModel(cv::String fileName, int nowEpoch);
};

