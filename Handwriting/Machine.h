#pragma once
#include "framework.h"

class Machine
{
public:
	//Ŀ�� ������ ���� : Ŀ�� ��, ��, ��
	std::vector<cv::Mat> kernels1;
	std::vector<cv::Mat> kernels2;

	cv::Size kernel1Stride;
	cv::Size kernel2Stride;

	//�ռ��� ������ ���� : ������ ��, ��, ��
	std::vector<cv::Mat> conv1;
	std::vector<cv::Mat> conv2;

	cv::Size poolSize;
	cv::Size poolStride;

	//�Ʒ� ������ ���� : ������ ��, ��, ��
	std::vector<cv::Mat> x1Mats;
	std::vector<cv::Mat> x2Mats;
	cv::Mat wMat;
	std::vector<cv::Mat> yHatMats;
	std::vector<cv::Mat> yMats;

	double lossAverage;
public:
	void Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec);
	//������ ��� �� yHat�� ��ȯ
	void ForwardPropagation(cv::InputArray _Input, cv::OutputArray _Output);
	void BackPropagation(cv::InputArray _Input, cv::OutputArray _Output);
	void Training(int epoch, float learningRate, float l2);

};

