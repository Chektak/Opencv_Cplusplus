#pragma once
#include "framework.h"

/// <summary>
/// 특성 스케일 0~1로 정규화
/// 전체 세트가 100이라면 훈련:검증:테스트 = 64:16:20 비율로 분리
/// 에포크에 따른 손실 함수, 정확도 그래프로 적절한 overfitting, underfitting 사이 점을 찾아(편향-분산 트레이드 오프) 훈련 조기 종료
/// </summary>
class CNNMachine
{
public:
	//훈련 데이터 순서 : 데이터 수, 행렬
	//(채널 수는 흑백 이미지 입력만을 가정하므로 생략)
	std::vector<cv::Mat> trainingMats; //훈련 세트 = 합성곱층 1 (Conv1) 입력 행렬
	std::vector<cv::Mat> validationMats; //검증 세트
	std::vector<cv::Mat> testMats; //테스트(시험) 세트
	cv::Mat trainingYMat;//훈련 정답(라벨) 행렬
	cv::Mat validationYMat;//검증 정답(라벨) 행렬
	cv::Mat testYMat;//테스트 정답(라벨) 행렬
	cv::Mat trainingNeuralYHatMat;//모델 훈련 예측행렬
	cv::Mat validationNeuralYHatMat;//모델 검증 예측행렬
	cv::Mat testNeuralYHatMat;//모델 테스트 예측행렬

#pragma region 정방향 계산에서 사용
	std::vector<cv::Mat> trainingConv1x1ZeroPaddingMats;//합성곱층 1 입력

	//데이터 순서 : 채널 수, 커널 수, 행렬
	std::vector<std::vector<cv::Mat>> kernels1;//합성곱층 1 입력
	std::vector<std::vector<cv::Mat>> kernels2;//합성곱층 2 입력
	std::vector<std::vector<double>> conv1ResultBiases;//합성곱층 1 결과에 더할 편향
	std::vector<std::vector<double>> conv2ResultBiases;//합성곱층 2 결과에 더할 편향

	//정방향 합성곱 계산 시 사용하는 행렬들
	//데이터 순서 : 데이터 수, 채널 수, 행렬
	std::vector<std::vector<cv::Mat>> trainingConv1ResultMats;//합성곱층 1 결과
	std::vector<std::vector<cv::Mat>> trainingConv2ResultMats;//합성곱층 2 결과
	std::vector<std::vector<cv::Mat>> trainingConv1ResultZeroPadMats;//풀링층 1 입력
	std::vector<std::vector<cv::Mat>> trainingConv2ResultZeroPadMats;//풀링층 2 입력

	//정방향 풀링 시 사용하는 행렬들
	//데이터 순서 : 데이터 수, 채널 수, 행렬
	std::vector<std::vector<cv::Mat>> trainingPool1Result;//풀링층 1 결과
	std::vector<std::vector<cv::Mat>> trainingPool1ResultZeroPadding;//합성곱층 2 입력
	std::vector<std::vector<cv::Mat>> trainingPool2Result;//풀링층 2 결과

	//정방향 완전연결신경망 계산 시 사용하는 행렬들
	cv::Mat trainingNeuralX1Mat;//완전연결신경망 1층 입력 (trainingPool2Result를 2차원으로 펼친 형태, 행 : 합성곱 1층 입력 데이터 수, 열 : 풀링층2 결과의 채널 수 * 행 * 열) 
	cv::Mat neuralW1Mat;//완전연결신경망 1층 입력
	std::vector<double> neuralBiases1; //완전연결신경망 1층 입력, trainingNeuralX1Mat의 행 수만큼의 bias
	cv::Mat trainingNeuralX2;//완전연결신경망 1층 결과, 완전연결신경망 2층 입력
	cv::Mat neuralW2;//완전연결신경망 2층 입력
	std::vector<double> neuralBiases2; //완전연결신경망 2층 입력, trainingNeuralX2의 행 수만큼의 bias
#pragma endregion

#pragma region 역방향 계산에서 사용
	//역방향 계산시 사용하는 Max풀링 필터(풀링을 입력행렬에 대해 미분)
	//데이터 순서 : 데이터 수, 커널 수, 행렬
	std::vector<std::vector<cv::Mat>> pool1BackpropFilters;
	std::vector<std::vector<cv::Mat>> pool2BackpropFilters;

	/* 합성곱을 커널에 대해 바로 미분할 수 없는 이유							*
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

#pragma region 저장할 모델 데이터(가중치 이외)
	cv::Size kernel1Stride;
	cv::Size kernel2Stride;
	cv::Size poolSize;
	cv::Size poolStride;
	cv::Size trainingPool1ResultSize;
	cv::Size trainingPool2ResultSize;

	std::vector<double> lossAverages;
	double loss = 0;
	double learningRate = 0;
	int nowEpoch = 0;

	int useData_Num = 0;
	int kernel1_Num = 0;
	int kernel2_Num = 0;
	int classification_Num = CLASSIFICATIONNUM;

	enum GD { STOCHASTIC, MINI_BATCH, BATCH };
	GD gradientDescent = GD::BATCH;
#pragma endregion

#pragma region Application 임시 변수
	int autoTrainingDelay = 0;
	bool autoTraining = false;

	cv::TickMeter trainingTickMeter;
	cv::TickMeter predictTickMeter;

	OpencvPractice* op;

	static cv::Point mousePt;
	static bool mouseLeftPress;
	static bool mouseRightPress;
#pragma endregion

#pragma region 학습된 모델로 예측할 때 사용
	std::vector<cv::Mat> predictConv1Mats;//합성곱층 1 결과
	std::vector<cv::Mat> predictConv2Mats;//합성곱층 2 결과
	std::vector<cv::Mat> predictConv1ZeroPaddingMats;//풀링층 1 입력
	std::vector<cv::Mat> predictConv2ZeroPaddingMats;//풀링층 2 입력

	//풀링 시 사용하는 행렬들
	//데이터 순서 : 채널 수, 행렬
	std::vector<cv::Mat> predictPool1Result;//풀링층 1 결과
	std::vector<cv::Mat> predictPool1ResultZeroPadding;//합성곱층 2 입력
	std::vector<cv::Mat> predictPool2Result;//풀링층 2 결과

	cv::Mat predictNeuralX1Mat;//완전연결신경망 1층 입력 (trainingPool2Result를 2차원으로 펼친 형태)
	cv::Mat predictNeuralX2Mat;//완전연결신경망 1층 결과, 완전연결신경망 2층 입력
	cv::Mat predictNeuralYHatMat;
#pragma endregion

public:
	
	void Training(int epoch, double learningRate, double l2, GD gradientDescent);
	void SplitData(const std::vector<cv::Mat> &mnistImageMats, const std::vector<uint8_t> &mnistImageLabels, const int& trainingSetRatio, const int& validationSetRatio, const int& testSetRatio);
	void Init(OpencvPractice* op, int useData_Num, int kernel1_Num, int kernel2_Num, int classification_Num);
	
	void ForwardPropagationStochastic(int trainingIndex);
	void BackPropagationStochastic(int trainingIndex);
	
	void ForwardPropagationMiniBatch(int miniBatchIndex);
	void BackPropagationMiniBatch(int miniBatchIndex);

	void ForwardPropagationBatch();
	void BackPropagationBatch();
	
	//합성곱 입력 크기 행렬을 입력으로 넣으면 가설 행렬을 아웃풋으로 반환
	void ModelPredict(cv::InputArray _Input, cv::OutputArray _NeuralYHatMatOutput);

	bool SaveModel(cv::String fileName);
	bool LoadModel(cv::String fileName);
	void ReleaseVectors();

	//훈련을 진행할경우 true, 진행하지 않을 경우 false 리턴
	bool KeyEvent(int key);

	void PaintWindow(cv::InputOutputArray paintMat, cv::String windowName, cv::Size windowSize, int exitAsciiCode);
	static void CallBackFunc(int event, int x, int y, int flags, void* userdata);
};
