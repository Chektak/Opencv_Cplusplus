#include "Math.h"

void Math::CreateZeroPadding(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& convResultSize, const cv::Size& k, const cv::Size& stride)
{
	cv::Mat input = _Input.getMat();
	_Input.copyTo(_Output);
	
	//�е� ���߱� �ڵ�ȭ(�ռ��� ��� ũ�� ��� https://excelsior-cjh.tistory.com/79)
	double p = 0;
	int oH = (int)((input.rows + 2 * p - k.height) / stride.height) + 1;
	int oW = (int)((input.cols + 2 * p - k.width) / stride.width) + 1;

	//���� �е� ��� ����
	//�е��� 0.5 �þ ������ ���� + ��, ������ + �Ʒ� ������ ��� Ȯ��
	while (oH != convResultSize.height) {
		p += 0.5;
		if (p - (int)p != 0)
			cv::copyMakeBorder(_Output, _Output, 1, 0, 0, 0, cv::BORDER_CONSTANT, 0);
			//ExpandMatrix(_Output, _Output, 1, 0, 0, 0);
		else
			cv::copyMakeBorder(_Output, _Output, 0, 1, 0, 0, cv::BORDER_CONSTANT, 0);
			//ExpandMatrix(_Output, _Output, 0, 1, 0, 0);

		oH = (int)((input.rows + 2 * p - k.height) / stride.height) + 1;
	}
	p = 0;
	while (oW != convResultSize.width) {
		p += 0.5;
		if (p - (int)p != 0)
			cv::copyMakeBorder(_Output, _Output, 0, 0, 1, 0, cv::BORDER_CONSTANT, 0);
			//ExpandMatrix(_Output, _Output, 0, 0, 1, 0);
		else
			cv::copyMakeBorder(_Output, _Output, 0, 0, 0, 1, cv::BORDER_CONSTANT, 0);
			//ExpandMatrix(_Output, _Output, 0, 0, 0, 1);

		oW = (int)((input.cols + 2 * p - k.width) / stride.width) + 1;
	}
}

void Math::ExpandMatrix(cv::InputArray _Input, cv::OutputArray _Output, int top, int bottom, int left, int right) {
	cv::Mat img = _Input.getMat();
	cv::Mat padded;
	padded.create(img.rows + top + bottom, img.cols + left + right, img.type());
	padded.setTo(0);

	img.copyTo(padded(cv::Rect(left, top, img.cols, img.rows)));
	padded.copyTo(_Output);
}
//template<typename T>
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
					output.at<float>(y, x) += kernel.at<float>(ky, kx)
						* zeroPaddingMat.at<float>(y * stride.height + ky, x * stride.width + kx);
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
	//std::cout << "Ǯ�� �Լ� �ȿ��� input size :\n" << outputWidth << std::endl<< outputHeight << std::endl;

	if (outputHeight != (int)outputHeight
		|| outputWidth != (int)outputWidth) {
		//std::cout << "Ǯ�� ���� �Ұ��ϹǷ� ���� �е� �߰�" << std::endl;
		outputHeight = cvRound(outputHeight);
		outputWidth = cvRound(outputWidth);
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, cv::Size(outputWidth, outputHeight),poolSize, stride);
	}
	_Output.create(cv::Size(outputWidth, outputHeight), zeroPaddingMat.type());
	_Output.setTo(0);
	cv::Mat output = _Output.getMat();
	
	/*std::cout << "Ǯ�� ���� �� input :\n" << zeroPaddingMat << std::endl;
	std::cout << "Ǯ�� ���� �� output :\n" << output << std::endl;
	std::cout << output.at<float>(0, 0) << std::endl;*/
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
			//std::cout << maxValue << std::endl;
			output.at<float>(y, x) = maxValue;
		}
	}
}



void Math::GetMaxPoolingFilter(cv::InputArray _PoolInput, cv::OutputArray _PoolFilter, cv::InputArray _PoolResult, const cv::Size& poolSize, const cv::Size& stride)
{
	cv::Mat poolInput = _PoolInput.getMat();
	cv::Mat poolResult = _PoolResult.getMat();
	
	//PoolFilter�� PoolInput�� ���� ũ��� ����
	_PoolFilter.create(_PoolInput.size(), _PoolInput.type());
	_PoolFilter.setTo(0);
	cv::Mat poolFilter = _PoolFilter.getMat();
	
	//poolResult ��� ��ҿ� �����ϴ� �Է� ��� ��Ҹ� ã��, Ǯ�� ���͸� Ȱ��ȭ
	for (int y = 0; y < poolResult.rows; y++) {
		for (int x = 0; x < poolResult.cols; x++) {
			bool findFlag = false;
			for (int ky = 0; ky < poolSize.height && findFlag == false; ky++) {
				for (int kx = 0; kx < poolSize.width; kx++) {
					if (poolResult.at<float>(y,x) == poolInput.at<float>(y * stride.height + ky, x * stride.width + kx)) {
						poolFilter.at<float>(y * stride.height + ky, x * stride.width + kx) = 1;
						findFlag = true;
						break;
					}
				}
			}
		}
	}
}

void Math::MaxPoolingReverse(cv::InputArray _Input, cv::OutputArray _Output, cv::InputArray _PoolFilter)
{
	cv::Mat inputMat = _Input.getMat();
	cv::Mat poolFilterMat = _PoolFilter.getMat();

	inputMat.copyTo(_Output);
	cv::Mat outputMat = _Output.getMat();

	outputMat = outputMat.mul(poolFilterMat);
}

void Math::GetConvolutionKFilters(cv::InputArray _Input, std::vector<std::vector<std::vector<float>>>* _Output, cv::InputArray k, const cv::Size& stride)
{
	//std::vector<std::vector<std::vector<float>>> _Output
	//������ ���� : k��� ũ���� ��, k��, �Է� ��İ�*k��*k��
	cv::Mat input = _Input.getMat();
	cv::Mat kernel = k.getMat();

	//Ŀ���� �ռ��� ����� ���� �е� ��Ŀ��� ������ Output�� ����
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			for (int ky = 0; ky < kernel.rows; ky++) {
				for (int kx = 0; kx < kernel.cols; kx++) {
					_Output->at(ky).at(kx).at((ky*(unsigned long long)kernel.cols)+kx) = input.at<float>(y * stride.height + ky, x * stride.width + kx);
				}
			}
		}
	}
}


void Math::ConvolutionReverse(cv::InputArray _ConvResult, cv::OutputArray _Output, cv::InputArray _ConvInput, const std::vector<std::vector<std::vector<float>>>& _KernelFilter)
{
	cv::Mat convResult = _ConvResult.getMat();
	_Output.create(cv::Size(_KernelFilter[0].size(), _KernelFilter.size()), CV_32FC1);
	_Output.setTo(0);
	cv::Mat output = _Output.getMat();

	for (int iY = 0; iY < convResult.rows; iY++) {
		for (int iX = 0; iX < convResult.cols; iX++) {

			for (int kY = 0; kY < _KernelFilter.size(); kY++) {
				for (int kX = 0; kX < _KernelFilter[0].size(); kX++) {
					/*float kElementSum = 0;

					for (int kN = 0; kN < _KernelFilter[0][0].size(); kN++) {
						kElementSum+=_KernelFilter[iY][iX][iY * _KernelFilter[0].size() + iX];
					}*/
					output.at<float>(kY, kX) =  / convResult.at<float>(iY, iX);
				}
			}
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
		sum = 0;
		max = 0;
		//���� ���� max�� ���Ѵ�
		for (int x = 0; x < input.cols; x++) {
			if (max < input.at<float>(y, x))
				max = input.at<float>(y, x);
		}
		for (int x = 0; x < input.cols; x++) {
			output.at<float>(y, x) = exp(input.at<float>(y, x) - max);
			sum += output.at<float>(y, x);
		}
		output.row(y) /= sum;
	}

}