#include "Math.h"

void Math::CreateZeroPadding(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, const cv::Size& k, const cv::Size& stride)
{
	cv::Mat input = _Input.getMat();
	_Input.copyTo(_Output);

	//�е� ���߱� �ڵ�ȭ(�ռ��� ��� ũ�� ��� https://excelsior-cjh.tistory.com/79)
	double p = 0;
	int oH = (int)((input.rows + 2 * p - k.height) / stride.height) + 1;
	int oW = (int)((input.cols + 2 * p - k.width) / stride.width) + 1;
	//���� �е� ��� ����
	//�е��� 0.5 �þ ������ ���� + ��, ������ + �Ʒ� ������ ��� Ȯ��
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

//���� ��� ���� ����
void Math::Convolution(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, cv::InputArray k, const cv::Size& stride)
{
	cv::Mat kernel = k.getMat();
	cv::Mat zeroPaddingMat = _Input.getMat();
	//���� �е� ����� ��ȿ���� �ʴٸ�(�ռ��� ���� �Ұ� ��)
	if ((zeroPaddingMat.rows - kernel.rows)/stride.height + 1 != outputSize.height 
		|| (zeroPaddingMat.cols - kernel.cols)/stride.width + 1 != outputSize.width) {
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, zeroPaddingMat.size(),kernel.size(), stride);
	}

	_Output.create(outputSize, _Input.type());
	_Output.setTo(0);
	cv::Mat output = _Output.getMat();

	//���� �е� ��İ� Ŀ�η� ���� ��� ���� ��, ���� ����� _Output ��Ŀ� ����
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

	//���� �е� ����� ��ȿ���� ������ �˻�(Ǯ�� ���� ����)
	float outputHeight = (float)(zeroPaddingMat.rows - poolSize.height) / stride.height + 1;
	float outputWidth = (float)(zeroPaddingMat.cols - poolSize.width) / stride.width + 1;
	std::cout << "Ǯ�� �Լ� �ȿ��� input size :\n" << outputWidth << std::endl<< outputHeight << std::endl;

	if (outputHeight != (int)outputHeight
		|| outputWidth != (int)outputWidth) {
		std::cout << "Ǯ�� ���� �Ұ��ϹǷ� ���� �е� �߰�" << std::endl;
		outputHeight = cvRound(outputHeight);
		outputWidth = cvRound(outputWidth);
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, cv::Size(outputWidth, outputHeight),poolSize, stride);
	}
	_Output.create(cv::Size(outputWidth, outputHeight), zeroPaddingMat.type());
	_Output.setTo(0);
	cv::Mat output = _Output.getMat();
	
	std::cout << "Ǯ�� ���� �� input :\n" << zeroPaddingMat << std::endl;
	std::cout << "Ǯ�� ���� �� output :\n" << output << std::endl;
	std::cout << output.at<float>(0, 0) << std::endl;
	//���� �е� ��İ� Ŀ�η� ���� ��� ���� ��, ���� ����� output ��Ŀ� ����
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

	//softmax�� ���Ѵ�� �߻����� �ʰ� max�� ���� ����ó���Ѵ�.
	//https://leedakyeong.tistory.com/entry/�عٴں���-�����ϴ�-������-����Ʈ�ƽ�-�Լ�-�����ϱ�-in-���̽�-softmax-in-python
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