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

	//������ �ռ��� ��� �� ����ϴ� ��ĵ�
	//������ ���� : ������ ��, ä�� ��, ���
	std::vector<std::vector<cv::Mat>> conv1Mats;//�ռ����� 1 ���
	std::vector<std::vector<cv::Mat>> conv2Mats;//�ռ����� 2 ���
	std::vector<std::vector<cv::Mat>> conv1ZeroPaddingMats;//Ǯ���� 1 �Է�
	std::vector<std::vector<cv::Mat>> conv2ZeroPaddingMats;//Ǯ���� 2 �Է�
	
	//������ Ǯ�� �� ����ϴ� ��ĵ�
	//������ ���� : ������ ��, ä�� ��, ���
	std::vector<std::vector<cv::Mat>> poolresult1;//Ǯ���� 1 ���
	std::vector<std::vector<cv::Mat>> poolresult1ZeroPadding;//�ռ����� 2 �Է�
	std::vector<std::vector<cv::Mat>> poolresult2;//Ǯ���� 2 ���
	cv::Size poolSize;
	cv::Size poolStride;
	cv::Size pool1ResultSize;
	cv::Size pool2ResultSize;

	//������ ��������Ű�� ��� �� ����ϴ� ��ĵ�
	cv::Mat xMat;//poolresult2�� �Ű�� �Է����� ��ģ ����
	cv::Mat wMat;
	cv::Mat yHatMat;
	cv::Mat yMat;

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

	cv::Mat yLoss; 
	cv::Mat wT;
	cv::Mat yLossW;

	double lossAverage;
	double cost;
public:
	void Training(int epoch, double learningRate, double l2);
	void Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec);
	//������ ���
	void ForwardPropagation();
	void BackPropagation(double learningRate);

};

