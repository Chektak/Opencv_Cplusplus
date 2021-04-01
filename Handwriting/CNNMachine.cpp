#include "CNNMachine.h"

void CNNMachine::Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec)
{
	trainingMats = &trainingVec;
#pragma region ��� ����ġ ����� �յ� ������ ���� �ʱ�ȭ
	cv::RNG gen(cv::getTickCount());
	//Ŀ�� 1�� ä�� �Ѱ�
	kernels1.push_back(std::vector<cv::Mat>());
	for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
		kernels1[0].push_back(cv::Mat(cv::Size(3, 3), CV_16FC1));
		gen.fill(kernels1[0][k1i], cv::RNG::UNIFORM, cv::Scalar(-10), cv::Scalar(9));

		kernels2.push_back(std::vector<cv::Mat>());
		//Ŀ�� 2�� ä���� Ŀ�� 1�� ����
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			kernels2[k1i].push_back(cv::Mat(cv::Size(3, 3), CV_16FC1));
			gen.fill(kernels2[k1i][k2i], cv::RNG::UNIFORM, cv::Scalar(-10), cv::Scalar(9));
		}
	}
	
	kernel1Stride = cv::Size(1,1);
	kernel2Stride = cv::Size(1,1);
	
	//�ռ����� ���� �е����� �����ϹǷ� Ǯ���� 2�������� ��Ҹ� ���
	int wHeight = ((*trainingMats)[0].rows - poolSize.height) / poolStride.height + 1;
		wHeight = (wHeight - poolSize.height) / poolStride.height + 1;
	int wWidth = ((*trainingMats)[0].cols - poolSize.width) / poolStride.width + 1;
		wWidth = (wWidth - poolSize.width) / poolStride.width + 1;

	wMat.create(cv::Size(wHeight*wWidth*KERNEL2_NUM, CLASSIFICATIONNUM), CV_16FC1);
	gen.fill(wMat, cv::RNG::UNIFORM, cv::Scalar(-10), cv::Scalar(9));
#pragma endregion
	for (int j = 0; j < (*trainingMats).size(); j++) {
		conv1Mats.push_back(std::vector<cv::Mat>());
		conv2Mats.push_back(std::vector<cv::Mat>());
		poolresult1.push_back(std::vector<cv::Mat>());
		poolresult2.push_back(std::vector<cv::Mat>());
		for (int i = 0; i < KERNEL1_NUM; i++) {
			conv1Mats[j][i].push_back(cv::Mat());
			poolresult1[j][i].push_back(cv::Mat());
		}
		for (int i = 0; i < KERNEL2_NUM; i++) {
			conv2Mats[j][i].push_back(cv::Mat());
			poolresult2[j][i].push_back(cv::Mat());
		}
	}
	//���� �����͸� ���ͷ� ��ȯ�Ѵ�.
	yMat.zeros(cv::Size(wHeight * wWidth * KERNEL2_NUM, CLASSIFICATIONNUM), CV_16FC1);
	for (int y = 0; y < labelVec.size(); y++) {
		for (int x = 0; x < CLASSIFICATIONNUM; x++) {
			//���� �´´ٸ� true(1), �ƴ϶�� false(0)�� ����
			if (labelVec[x] == x)
				yMat.at<float>(y, x) = 1;
		}
	}

	yHatMat.create(cv::Size(wHeight * wWidth * KERNEL2_NUM, CLASSIFICATIONNUM), CV_16FC1);
	poolSize = cv::Size(2, 2);
	poolStride = cv::Size(2, 2);

	lossAverage = 2305843009213693951;
}

void CNNMachine::ForwardPropagation()
{
	//��� ������
	cv::Size trainingMatrixSize = (*trainingMats)[0].size();
	cv::Size k1MatrixSize = kernels1[0][0].size();
	cv::Size k2MatrixSize = kernels2[0][0].size();

	//��������Ű�� �Է�
	xMat.create(cv::Size(0, wMat.rows), CV_16FC1);

	for (int x1i = 0; x1i < trainingMats->size(); x1i++) {
		Math::CreateZeroPadding((*trainingMats)[x1i], x1ZeroPaddingMats[x1i], trainingMatrixSize, k1MatrixSize, kernel1Stride);
		//�ռ����� 1
		for (int k1i = 0; k1i < conv1Mats[k1i].size(); k1i++) {
			Math::Convolution(x1ZeroPaddingMats[x1i], conv1Mats[x1i][k1i], trainingMatrixSize, kernels1[x1i][k1i], kernel1Stride);
			Math::CreateZeroPadding(conv1Mats[x1i][k1i], conv1ZeroPaddingMats[x1i][k1i], trainingMatrixSize, poolSize, poolStride);
			Math::MaxPooling(conv1ZeroPaddingMats[x1i][k1i], poolresult1[x1i][k1i], poolSize, poolStride);
			
			//�ռ����� 2
			/*�ռ����� 1�� (��:������ ��, ��:ä�� ��)�� �̹����� ���� poolresult1��İ� 
			�ռ����� 2�� kernel2����� ��İ��ϵ� ����*/
			for (int k2i = 0; k2i < conv2Mats[x1i].size(); k2i++) {
				Math::CreateZeroPadding(poolresult1[x1i][k1i], poolresultZeroPadding1[x1i][k1i], poolresult1[0][0].size(), k2MatrixSize, kernel2Stride);
				Math::Convolution(poolresultZeroPadding1[x1i][k1i], conv2Mats[x1i][k2i], poolresult1[0][0].size(), kernels2[k1i][k2i], kernel2Stride);
				Math::CreateZeroPadding(conv2Mats[x1i][k2i], conv2ZeroPaddingMats[x1i][k2i], poolresult1[0][0].size(), poolSize, poolStride);
				Math::MaxPooling(conv2ZeroPaddingMats[x1i][k2i], poolresult2[x1i][k2i], poolSize, poolStride);
			}
		}
		xMat.push_back(poolresult2[x1i].data());
	}

	//poolresult2�� ��ķ�
	//vec<vec<float>> ��� ��ȯ : https://stackoverflow.com/questions/18519647/opencv-convert-vector-of-vector-to-mat
	//Mat to float ��ȯ : https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv

}

void CNNMachine::Training(int epoch, float learningRate, float l2)
{
	for (int i = 0; i < epoch; i++) {
		//ForwardPropagation(x1Mats[x], yHatMat);



	}
}
