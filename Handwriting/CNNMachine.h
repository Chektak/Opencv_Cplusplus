#pragma once
#include "framework.h"

class CNNMachine
{
public:
	//�Ʒ� ������ ���� : ������ ��, ��, ��
	std::vector<cv::Mat> trainingMats; //=x1Mats
	std::vector<cv::Mat> x1ZeroPaddingMats;

	//Ŀ�� ������ ���� : ä�� ��, Ŀ�� ��, ��, ��
	std::vector<std::vector<cv::Mat>> kernels1;
	std::vector<std::vector<cv::Mat>> kernels2;
	cv::Size kernel1Stride;
	cv::Size kernel2Stride;

	//�ռ��� ������ ���� : ������ ��, ä�� ��, ��, ��
	std::vector<std::vector<cv::Mat>> conv1Mats;
	std::vector<std::vector<cv::Mat>> conv1ZeroPaddingMats;
	std::vector<std::vector<cv::Mat>> conv2Mats;
	std::vector<std::vector<cv::Mat>> conv2ZeroPaddingMats;
	
	//Ǯ����� ������ ���� : ������ ��, ä�� ��, ��, ��
	std::vector<std::vector<cv::Mat>> poolresult1;
	std::vector<std::vector<cv::Mat>> poolresult1ZeroPadding;
	std::vector<std::vector<cv::Mat>> poolresult2;
	cv::Size poolSize;
	cv::Size poolStride;

	//��������Ű�� �������� ��ķθ� ���
	cv::Mat xMat;//=poolresult2�� ��ģ ����
	cv::Mat wMat;
	cv::Mat yHatMat;
	cv::Mat yMat;

	//������ ���� ����ϱ� ���� conv1�� MaxǮ���� �� ���͸� ����
	std::vector<std::vector<cv::Mat>> conv1PoolFilters;
	std::vector<std::vector<cv::Mat>> conv2PoolFilters;

	double lossAverage;
public:
	void Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec);
	//������ ���
	void ForwardPropagation();
	void BackPropagation();
	void Training(int epoch, float learningRate, float l2);

};

