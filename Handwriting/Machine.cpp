#include "Machine.h"

void Machine::Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec)
{

	//모든 가중치 행렬을 균등 분포로 랜덤 초기화
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
	
	//합성곱은 세임 패딩으로 진행하므로 풀링층 2개에서의 축소만 계산
	int wHeight = (x1Mats[0].rows - poolSize.height) / poolStride.height + 1;
		wHeight = (wHeight - poolSize.height) / poolStride.height + 1;
	int wWidth = (x1Mats[0].cols - poolSize.width) / poolStride.width + 1;
		wWidth = (wWidth - poolSize.width) / poolStride.width + 1;

	wMat.create(cv::Size(wHeight*wWidth*KERNEL2_NUM, CLASSIFICATIONNUM), CV_16FC1);
	gen.fill(wMat, cv::RNG::UNIFORM, cv::Scalar(-10), cv::Scalar(9));

	//정답 데이터 벡터를 행렬로 변환한다.
	for (int i = 0; i < USEDATA_NUM; i++) {
		yMats[i].zeros(cv::Size(wHeight * wWidth * KERNEL2_NUM, CLASSIFICATIONNUM), CV_16FC1);
		for (int y = 0; y < labelVec.size(); y++) {
			for (int x = 0; x < CLASSIFICATIONNUM; x++) {
				//열과 맞는다면 true(1), 아니라면 false(0)를 저장
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
