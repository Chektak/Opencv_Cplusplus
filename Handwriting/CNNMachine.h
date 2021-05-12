#pragma once
#include "framework.h"

/// <summary>
/// Ư�� ������ 0~1�� ����ȭ
/// ��ü ��Ʈ�� 100�̶�� �Ʒ�:����:�׽�Ʈ = 64:16:20 ������ �и�
/// ����ũ�� ���� �ս� �Լ�, ��Ȯ�� �׷����� ������ overfitting, underfitting ���� ���� ã��(����-�л� Ʈ���̵� ����) �Ʒ� ���� ����
/// </summary>
class CNNMachine
{
public:
	//�Ʒ� ������ ���� : ������ ��, ���
	//(ä�� ���� ��� �̹��� �Է¸��� �����ϹǷ� ����)
	std::vector<cv::Mat> trainingMats; //�Ʒ� ��Ʈ = �ռ����� 1 (Conv1) �Է� ���
	std::vector<cv::Mat> validationMats; //���� ��Ʈ
	std::vector<cv::Mat> testMats; //�׽�Ʈ(����) ��Ʈ
	cv::Mat trainingYMat;//�Ʒ� ����(��) ���
	cv::Mat validationYMat;//���� ����(��) ���
	cv::Mat testYMat;//�׽�Ʈ ����(��) ���
	cv::Mat trainingNeuralYHatMat;//�� �Ʒ� �������
	cv::Mat validationNeuralYHatMat;//�� ���� �������
	cv::Mat testNeuralYHatMat;//�� �׽�Ʈ �������

#pragma region ������ ��꿡�� ���
	std::vector<cv::Mat> trainingConv1x1ZeroPaddingMats;//�ռ����� 1 �Է�

	//������ ���� : ä�� ��, Ŀ�� ��, ���
	std::vector<std::vector<cv::Mat>> kernels1;//�ռ����� 1 �Է�
	std::vector<std::vector<cv::Mat>> kernels2;//�ռ����� 2 �Է�
	std::vector<std::vector<double>> conv1ResultBiases;//�ռ����� 1 ����� ���� ����
	std::vector<std::vector<double>> conv2ResultBiases;//�ռ����� 2 ����� ���� ����

	//������ �ռ��� ��� �� ����ϴ� ��ĵ�
	//������ ���� : ������ ��, ä�� ��, ���
	std::vector<std::vector<cv::Mat>> trainingConv1ResultMats;//�ռ����� 1 ���
	std::vector<std::vector<cv::Mat>> trainingConv2ResultMats;//�ռ����� 2 ���
	std::vector<std::vector<cv::Mat>> trainingConv1ResultZeroPadMats;//Ǯ���� 1 �Է�
	std::vector<std::vector<cv::Mat>> trainingConv2ResultZeroPadMats;//Ǯ���� 2 �Է�

	//������ Ǯ�� �� ����ϴ� ��ĵ�
	//������ ���� : ������ ��, ä�� ��, ���
	std::vector<std::vector<cv::Mat>> trainingPool1Result;//Ǯ���� 1 ���
	std::vector<std::vector<cv::Mat>> trainingPool1ResultZeroPadding;//�ռ����� 2 �Է�
	std::vector<std::vector<cv::Mat>> trainingPool2Result;//Ǯ���� 2 ���

	//������ ��������Ű�� ��� �� ����ϴ� ��ĵ�
	cv::Mat trainingNeuralX1Mat;//��������Ű�� 1�� �Է� (trainingPool2Result�� 2�������� ��ģ ����, �� : �ռ��� 1�� �Է� ������ ��, �� : Ǯ����2 ����� ä�� �� * �� * ��) 
	cv::Mat neuralW1Mat;//��������Ű�� 1�� �Է�
	std::vector<double> neuralBiases1; //��������Ű�� 1�� �Է�, trainingNeuralX1Mat�� �� ����ŭ�� bias
	cv::Mat trainingNeuralX2;//��������Ű�� 1�� ���, ��������Ű�� 2�� �Է�
	cv::Mat neuralW2;//��������Ű�� 2�� �Է�
	std::vector<double> neuralBiases2; //��������Ű�� 2�� �Է�, trainingNeuralX2�� �� ����ŭ�� bias
#pragma endregion

#pragma region ������ ��꿡�� ���
	//������ ���� ����ϴ� MaxǮ�� ����(Ǯ���� �Է���Ŀ� ���� �̺�)
	//������ ���� : ������ ��, Ŀ�� ��, ���
	std::vector<std::vector<cv::Mat>> pool1BackpropFilters;
	std::vector<std::vector<cv::Mat>> pool2BackpropFilters;

	/* �ռ����� Ŀ�ο� ���� �ٷ� �̺��� �� ���� ����							*
	*	: ���� �е� ���̳� ���� 0�� �Է������ ������ �� ����					*
	*	�ذ�å1 : �Է������ �����е��� �� ���� �Է������ st, ed ����Ʈ�� ����	*
	*	�ذ�å2 : �Է���İ� �����е�, ��Ʈ���̵�, Ŀ�� ũ�⸦ �м���			*
	*			�������� �����е� �κ��� �˾Ƴ���							*
	*   �ذ�å 1 ���														*/

	//������ ���� ����ϴ� �ռ��� ����(���� �е��� �Է������ �����ϱ� ���� ��ǥ�� ����� ���)
	//������ ���� : �ռ��� ������ ��*��, pair(Ŀ�� ���� x �Է� ��� start index, Ŀ�� ���� x �Է� ��� end index)
	std::vector<std::pair<int, int>> conv1BackpropFilters;
	std::vector<std::pair<int, int>> conv2BackpropFilters;

	cv::Mat yLoss; //�ս��Լ��� SoftMax �Լ� ����� ���� �̺��� ��
	cv::Mat w2T;
	cv::Mat yLossW2;//�ս� �Լ��� ����������2 �Է�(ReLu(aMat))�� ���� �̺��� ��
	cv::Mat yLossW2Relu3; //�ս��Լ��� ����������1 ����� ���� �̺��� ��
	cv::Mat yLossW2Relu3W1; //�ս� �Լ��� ����������1 �Է¿� ���� �̺��� ��

	std::vector<std::vector<cv::Mat>> yLossW2Relu3W1UpRelu2; //�ս� �Լ��� �ռ���2 ����� ���� �̺��� �� (Up�� Up-Sampleling(Ǯ���Լ��� �̺�))
	std::vector<std::vector<cv::Mat>> yLossW2Relu3W1UpRelu2P1UpRelu; //�ս� �Լ��� �ռ���1 ����� ���� �̺��� ��
#pragma endregion

#pragma region ������ �� ������(����ġ �̿�)
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

#pragma region Application �ӽ� ����
	int autoTrainingDelay = 0;
	bool autoTraining = false;

	cv::TickMeter trainingTickMeter;
	cv::TickMeter predictTickMeter;

	OpencvPractice* op;

	static cv::Point mousePt;
	static bool mouseLeftPress;
	static bool mouseRightPress;
#pragma endregion

#pragma region �н��� �𵨷� ������ �� ���
	std::vector<cv::Mat> predictConv1Mats;//�ռ����� 1 ���
	std::vector<cv::Mat> predictConv2Mats;//�ռ����� 2 ���
	std::vector<cv::Mat> predictConv1ZeroPaddingMats;//Ǯ���� 1 �Է�
	std::vector<cv::Mat> predictConv2ZeroPaddingMats;//Ǯ���� 2 �Է�

	//Ǯ�� �� ����ϴ� ��ĵ�
	//������ ���� : ä�� ��, ���
	std::vector<cv::Mat> predictPool1Result;//Ǯ���� 1 ���
	std::vector<cv::Mat> predictPool1ResultZeroPadding;//�ռ����� 2 �Է�
	std::vector<cv::Mat> predictPool2Result;//Ǯ���� 2 ���

	cv::Mat predictNeuralX1Mat;//��������Ű�� 1�� �Է� (trainingPool2Result�� 2�������� ��ģ ����)
	cv::Mat predictNeuralX2Mat;//��������Ű�� 1�� ���, ��������Ű�� 2�� �Է�
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
	
	//�ռ��� �Է� ũ�� ����� �Է����� ������ ���� ����� �ƿ�ǲ���� ��ȯ
	void ModelPredict(cv::InputArray _Input, cv::OutputArray _NeuralYHatMatOutput);

	bool SaveModel(cv::String fileName);
	bool LoadModel(cv::String fileName);
	void ReleaseVectors();

	//�Ʒ��� �����Ұ�� true, �������� ���� ��� false ����
	bool KeyEvent(int key);

	void PaintWindow(cv::InputOutputArray paintMat, cv::String windowName, cv::Size windowSize, int exitAsciiCode);
	static void CallBackFunc(int event, int x, int y, int flags, void* userdata);
};
