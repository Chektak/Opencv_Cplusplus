#pragma once
#include "framework.h"

class CNNMachine
{
public:
	//�Ʒ� ������ ���� : ������ ��, ���
	//(ä�� ���� ��� �̹��� �Է¸��� �����ϹǷ� ����)
	std::vector<cv::Mat> trainingMats; //=x1Mats
	std::vector<cv::Mat> x1ZeroPaddingMats;//�ռ����� 1 �Է�

	//Ŀ�� ������ ���� : ä�� ��, Ŀ�� ��, ���
	std::vector<std::vector<cv::Mat>> kernels1;//�ռ����� 1 �Է�
	std::vector<std::vector<cv::Mat>> kernels2;//�ռ����� 2 �Է�
	cv::Size kernel1Stride;
	cv::Size kernel2Stride;

	//������ ���� : ������ ��, ä�� ��, ��Į��
	std::vector<std::vector<double>> conv1Bias;//�ռ����� 1 ����� ���� ����
	std::vector<std::vector<double>> conv2Bias;//�ռ����� 2 ����� ���� ����
	
	//������ �ռ��� ��� �� ����ϴ� ��ĵ�
	//������ ���� : ������ ��, ä�� ��, ���
	std::vector<std::vector<cv::Mat>> conv1Mats;//�ռ����� 1 ���
	std::vector<std::vector<cv::Mat>> conv2Mats;//�ռ����� 2 ���
	std::vector<std::vector<cv::Mat>> conv1ZeroPaddingMats;//Ǯ���� 1 �Է�
	std::vector<std::vector<cv::Mat>> conv2ZeroPaddingMats;//Ǯ���� 2 �Է�
	
	//������ Ǯ�� �� ����ϴ� ��ĵ�
	//������ ���� : ������ ��, ä�� ��, ���
	std::vector<std::vector<cv::Mat>> pool1result;//Ǯ���� 1 ���
	std::vector<std::vector<cv::Mat>> pool1resultZeroPadding;//�ռ����� 2 �Է�
	std::vector<std::vector<cv::Mat>> pool2result;//Ǯ���� 2 ���
	cv::Size poolSize;
	cv::Size poolStride;
	cv::Size pool1ResultSize;
	cv::Size pool2ResultSize;

	//������ ��������Ű�� ��� �� ����ϴ� ��ĵ�
	cv::Mat xMat;//��������Ű�� 1�� �Է� (pool2result�� 2�������� ��ģ ����)
	cv::Mat w1Mat;//��������Ű�� 1�� �Է�
	double bias1 = 0;
	cv::Mat a1Mat;//��������Ű�� 1�� ���, ��������Ű�� 2�� �Է�
	cv::Mat w2Mat;//��������Ű�� 2�� �Է�
	double bias2 = 0;
	cv::Mat yHatMat;//�� ������
	cv::Mat yMat;//���� ���

#pragma region ������ ��꿡���� ���
	//������ ���� ����ϴ� MaxǮ�� ����(Ǯ���� �Է���Ŀ� ���� �̺�)
	//������ ���� : ������ ��, Ŀ�� ��, ���
	std::vector<std::vector<cv::Mat>> pool1BackpropFilters;
	std::vector<std::vector<cv::Mat>> pool2BackpropFilters;

	/* �ռ����� Ŀ�ο� ���� �̺��� �� ���� ����								*
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
	double loss;
	double learningRate;
public:
	void Training(int epoch, double learningRate, double l2);
	void Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec);
	//������ ���
	void ForwardPropagation();
	void BackPropagation(double learningRate);
	void SaveModel(cv::String fileName, int nowEpoch);
};

