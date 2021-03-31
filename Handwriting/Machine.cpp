#include "Machine.h"

void Machine::Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec)
{

	//��� ����ġ ����� �յ� ������ ���� �ʱ�ȭ
	cv::RNG gen(cv::getTickCount());

	for (int i = 0; i < KERNEL1_NUM; i++) {
		kernels1.push_back(cv::Mat(cv::Size(3, 3), CV_16FC1));
		gen.fill(kernels1[i], cv::RNG::UNIFORM, cv::Scalar(-10), cv::Scalar(9));

	}
	for (int i = 0; i < KERNEL2_NUM; i++) {
		kernels2.push_back(cv::Mat(cv::Size(3, 3), CV_16FC1));
		gen.fill(kernels2[i], cv::RNG::UNIFORM, cv::Scalar(-10), cv::Scalar(9));
	}

	kernel1Stride = cv::Size(1,1);
	kernel2Stride = cv::Size(1,1);
	
	//�ռ����� ���� �е����� �����ϹǷ� Ǯ���� 2�������� ��Ҹ� ���
	int wHeight = (x1Mats[0].rows - poolSize.height) / poolStride.height + 1;
		wHeight = (wHeight - poolSize.height) / poolStride.height + 1;
	int wWidth = (x1Mats[0].cols - poolSize.width) / poolStride.width + 1;
		wWidth = (wWidth - poolSize.width) / poolStride.width + 1;

	wMat.create(cv::Size(wHeight*wWidth*KERNEL2_NUM, CLASSIFICATIONNUM), CV_16FC1);
	gen.fill(wMat, cv::RNG::UNIFORM, cv::Scalar(-10), cv::Scalar(9));

	//���� ������ ���͸� ��ķ� ��ȯ�Ѵ�.
	for (int i = 0; i < USEDATA_NUM; i++) {
		yMats[i].zeros(cv::Size(wHeight * wWidth * KERNEL2_NUM, CLASSIFICATIONNUM), CV_16FC1);
		for (int y = 0; y < labelVec.size(); y++) {
			for (int x = 0; x < CLASSIFICATIONNUM; x++) {
				//���� �´´ٸ� true(1), �ƴ϶�� false(0)�� ����
				if(labelVec[i] == x)
				yMats[i].at<float>(y, x) = 
			}
		}
	}
	poolSize = cv::Size(2, 2);
	poolStride = cv::Size(2, 2);

	lossAverage = 2305843009213693951;
}

void Machine::Training(int epoch, float learningRate, float l2)
{
	for (int i = 0; i < epoch; i++) {

	}
}
