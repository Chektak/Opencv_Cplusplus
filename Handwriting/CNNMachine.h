#pragma once
#include "framework.h"

class CNNMachine
{
public:
	//�Ʒ� ������ ���� : ������ ��, ���
	//(ä�� ���� ��� �̹��� �Է¸��� �����ϹǷ� ����)
	std::vector<cv::Mat> trainingMats; //=x1Mats
	std::vector<cv::Mat> x1ZeroPaddingMats;//�ռ����� 1 �Է�

	//������ ���� : ä�� ��, Ŀ�� ��, ���
	std::vector<std::vector<cv::Mat>> kernels1;//�ռ����� 1 �Է�
	std::vector<std::vector<cv::Mat>> kernels2;//�ռ����� 2 �Է�
	cv::Size kernel1Stride;
	cv::Size kernel2Stride;
	std::vector<std::vector<double>> conv1ResultBiases;//�ռ����� 1 ����� ���� ����
	std::vector<std::vector<double>> conv2ResultBiases;//�ռ����� 2 ����� ���� ����

	
	//������ �ռ��� ��� �� ����ϴ� ��ĵ�
	//������ ���� : ������ ��, ä�� ��, ���
	std::vector<std::vector<cv::Mat>> conv1ResultMats;//�ռ����� 1 ���
	std::vector<std::vector<cv::Mat>> conv2ResultMats;//�ռ����� 2 ���
	std::vector<std::vector<cv::Mat>> conv1ResultZeroPadMats;//Ǯ���� 1 �Է�
	std::vector<std::vector<cv::Mat>> conv2ResultZeroPadMats;//Ǯ���� 2 �Է�
	
	//������ Ǯ�� �� ����ϴ� ��ĵ�
	//������ ���� : ������ ��, ä�� ��, ���
	std::vector<std::vector<cv::Mat>> pool1Result;//Ǯ���� 1 ���
	std::vector<std::vector<cv::Mat>> pool1ResultZeroPadding;//�ռ����� 2 �Է�
	std::vector<std::vector<cv::Mat>> pool2Result;//Ǯ���� 2 ���
	cv::Size poolSize;
	cv::Size poolStride;
	cv::Size pool1ResultSize;
	cv::Size pool2ResultSize;

	//������ ��������Ű�� ��� �� ����ϴ� ��ĵ�
	cv::Mat xMat;//��������Ű�� 1�� �Է� (pool2Result�� 2�������� ��ģ ����, �� : �ռ��� 1�� �Է� ������ ��, �� : Ǯ����2 ����� ä�� �� * �� * ��) 
	cv::Mat w1Mat;//��������Ű�� 1�� �Է�
	std::vector<double> biases1; //��������Ű�� 1�� �Է�, xMat�� �� ����ŭ�� bias
	cv::Mat a1Mat;//��������Ű�� 1�� ���, ��������Ű�� 2�� �Է�
	cv::Mat w2Mat;//��������Ű�� 2�� �Է�
	std::vector<double> biases2; //��������Ű�� 2�� �Է�, a1Mat�� �� ����ŭ�� bias
	cv::Mat yHatMat;//�� ������
	cv::Mat yMat;//���� ���

#pragma region ������ ��꿡���� ���
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

	std::vector<double> lossAverages;  
	double loss = 0;
	double learningRate = 0;
	int nowEpoch = 0;

	int useData_Num = 0;
	int kernel1_Num = 0;
	int kernel2_Num = 0;
	int classification_Num = CLASSIFICATIONNUM;

	int autoTrainingDelay = 0;
	bool autoTraining = false;
	cv::TickMeter trainingTickMeter;
	cv::TickMeter predictTickMeter;

	OpencvPractice* op;

	static cv::Point mousePt;
	static bool mouseLeftPress;
	static bool mouseRightPress;

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

	cv::Mat predictXMat;//��������Ű�� 1�� �Է� (pool2Result�� 2�������� ��ģ ����)
	cv::Mat predictA1Mat;//��������Ű�� 1�� ���, ��������Ű�� 2�� �Է�
	cv::Mat predictYHatMat;
#pragma endregion

public:
	void Training(int epoch, double learningRate, double l2);
	void Init(OpencvPractice* op, int useData_Num, int kernel1_Num, int kernel2_Num, int classification_Num);
	//������ ���
	void ForwardPropagation();
	void BackPropagation();
	void ModelPredict(cv::InputArray _Input);

	bool SaveModel(cv::String fileName);
	bool LoadModel(cv::String fileName);
	void ReleaseVectors();

	//�Ʒ��� �����Ұ�� true, �������� ���� ��� false ����
	bool KeyEvent(int key);

	void PaintWindow(cv::InputOutputArray paintMat, cv::String windowName, cv::Size windowSize, int exitAsciiCode);
	static void CallBackFunc(int event, int x, int y, int flags, void* userdata);
};
