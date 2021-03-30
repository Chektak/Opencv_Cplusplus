#include "Math.h"

void Math::CreateZeroPadding(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, const cv::Size& k, const cv::Size& stride)
{
	cv::Mat input = _Input.getMat();
	_Input.copyTo(_Output);

	//패딩 맞추기 자동화(합성곱 출력 크기 계산 https://excelsior-cjh.tistory.com/79)
	double p = 0;
	int oH = (int)((input.rows + 2 * p - k.height) / stride.height) + 1;
	int oW = (int)((input.cols + 2 * p - k.width) / stride.width) + 1;
	//제로 패딩 행렬 생성
	//패딩이 0.5 늘어날 때마다 왼쪽 + 위, 오른쪽 + 아래 순으로 행렬 확장
	while (oH != outputSize.height) {
		p += 0.5;
		if (p - (int)p != 0)
			cv::copyMakeBorder(_Output, _Output, 1, 0, 0, 0, cv::BORDER_CONSTANT, 0);
		else
			cv::copyMakeBorder(_Output, _Output, 0, 1, 0, 0, cv::BORDER_CONSTANT, 0);

		oH = (int)((input.rows + 2 * p - k.height) / stride.height) + 1;
	}
	p = 0;
	while (oW != outputSize.width) {
		p += 0.5;
		if (p - (int)p != 0)
			cv::copyMakeBorder(_Output, _Output, 0, 0, 1, 0, cv::BORDER_CONSTANT, 0);
		else
			cv::copyMakeBorder(_Output, _Output, 0, 0, 0, 1, cv::BORDER_CONSTANT, 0);

		oW = (int)((input.cols + 2 * p - k.width) / stride.width) + 1;
	}
}

//교차 상관 연산 수행
void Math::Convolution(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, cv::InputArray k, const cv::Size& stride)
{
	cv::Mat kernel = k.getMat();
	cv::Mat zeroPaddingMat = _Input.getMat();
	//제로 패딩 행렬이 유효하지 않다면(합성곱 성립 불가 시)
	if ((zeroPaddingMat.rows - kernel.rows)/stride.height + 1 != outputSize.height 
		|| (zeroPaddingMat.cols - kernel.cols)/stride.width + 1 != outputSize.width) {
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, zeroPaddingMat.size(),kernel.size(), stride);
	}

	_Output.create(outputSize, _Input.type());
	_Output.setTo(0);
	cv::Mat output = _Output.getMat();

	//제로 패딩 행렬과 커널로 교차 상관 연산 후, 연산 결과를 _Output 행렬에 저장
	for (int y = 0; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x ++) {
			for (int ky = 0; ky < kernel.rows; ky++) {
				for (int kx = 0; kx < kernel.cols; kx++) {
					output.at<float>(y, x) += kernel.at<float>(ky, kx) * zeroPaddingMat.at<float>(y * stride.height + ky, x * stride.width + kx);
				}
			}
		}
	}
}

void Math::Relu(cv::InputArray _Input, cv::OutputArray _Output)
{
	_Input.getMat().copyTo(_Output);
	cv::Mat output = _Output.getMat();
	for (int y = 0; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {
			if(output.at<float>(y, x) < 0)
				output.at<float>(y, x) = 0;
		}
	}
}

void Math::MaxPooling(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& poolSize, const cv::Size& stride)
{
	cv::Mat zeroPaddingMat = _Input.getMat();

	//제로 패딩 행렬이 유효하지 않은지 검사(풀링 성립 유무)
	float outputHeight = (float)(zeroPaddingMat.rows - poolSize.height) / stride.height + 1;
	float outputWidth = (float)(zeroPaddingMat.cols - poolSize.width) / stride.width + 1;
	std::cout << "풀링 함수 안에서 input size :\n" << outputWidth << std::endl<< outputHeight << std::endl;

	if (outputHeight != (int)outputHeight
		|| outputWidth != (int)outputWidth) {
		std::cout << "풀링 연산 불가하므로 제로 패딩 추가" << std::endl;
		outputHeight = cvRound(outputHeight);
		outputWidth = cvRound(outputWidth);
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, cv::Size(outputWidth, outputHeight),poolSize, stride);
	}
	_Output.create(cv::Size(outputWidth, outputHeight), zeroPaddingMat.type());
	_Output.setTo(0);
	cv::Mat output = _Output.getMat();
	
	std::cout << "풀링 연산 전 input :\n" << zeroPaddingMat << std::endl;
	std::cout << "풀링 연산 전 output :\n" << output << std::endl;
	std::cout << output.at<float>(0, 0) << std::endl;
	//제로 패딩 행렬과 커널로 교차 상관 연산 후, 연산 결과를 output 행렬에 저장
	for (int y = 0; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {
			float maxValue = 0;
			for (int ky = 0; ky < poolSize.height; ky++) {
				for (int kx = 0; kx < poolSize.width; kx++) {
					if (maxValue < zeroPaddingMat.at<float>(y * stride.height + ky, x * stride.width + kx))
						maxValue = zeroPaddingMat.at<float>(y * stride.height + ky, x * stride.width + kx);
				}
			}
			std::cout << maxValue << std::endl;
			output.at<float>(y, x) = maxValue;
		}
	}
}

void Math::NeuralNetwork(cv::InputArray _Input, cv::OutputArray _Output, cv::InputArray w)
{
	cv::Mat output = _Input.getMat();
	cv::Mat wMat = w.getMat();
	
	output *= wMat;
	output.copyTo(_Output);
}

void Math::SoftMax(cv::InputArray _Input, cv::OutputArray _Output)
{
	cv::Mat input = _Input.getMat();
	_Output.create(input.size(), input.type());

	cv::Mat output = _Output.getMat();

	long double sum = 0;
	float max = 0;

	//softmax가 무한대로 발산하지 않게 max를 구해 예외처리한다.
	//https://leedakyeong.tistory.com/entry/밑바닥부터-시작하는-딥러닝-소프트맥스-함수-구현하기-in-파이썬-softmax-in-python
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			if (max < input.at<float>(y, x))
				max = input.at<float>(y, x);
		}
	}

	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			output.at<float>(y, x) = exp(input.at<float>(y, x) - max);
			sum += output.at<float>(y, x);
		}
	}
	output /= sum;
}