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
					output.at<double>(y, x) += kernel.at<double>(ky, kx)
						* zeroPaddingMat.at<double>(y * stride.height + ky, x * stride.width + kx);
				}
			}
		}
	}
	//����� ����� ������ Ư�� ������ ǥ��ȭ
	output /= kernel.rows * kernel.cols;
}

void Math::Relu(cv::InputArray _Input, cv::OutputArray _Output)
{
	_Input.getMat().copyTo(_Output);
	cv::Mat output = _Output.getMat();
	for (int y = 0; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {
			if(output.at<double>(y, x) < 0)
				output.at<double>(y, x) = 0;
		}
	}
}

void Math::MaxPooling(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& poolSize, const cv::Size& stride)
{
	cv::Mat zeroPaddingMat = _Input.getMat();

	//���� �е� ����� ��ȿ���� ������ �˻�(Ǯ�� ���� ����)
	double outputHeight = (double)(zeroPaddingMat.rows - poolSize.height) / stride.height + 1;
	double outputWidth = (double)(zeroPaddingMat.cols - poolSize.width) / stride.width + 1;
	//std::cout << "Ǯ�� �Լ� �ȿ��� input size :\n" << outputWidth << std::endl<< outputHeight << std::endl;

	if (outputHeight != (int)outputHeight
		|| outputWidth != (int)outputWidth) {
		//std::cout << "Ǯ�� ���� �Ұ��ϹǷ� ���� �е� �߰�" << std::endl;
		outputHeight = cvRound(outputHeight);
		outputWidth = cvRound(outputWidth);
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, cv::Size((int)outputWidth, (int)outputHeight),poolSize, stride);
	}
	_Output.create(cv::Size((int)outputWidth, (int)outputHeight), zeroPaddingMat.type());
	_Output.setTo(0);
	cv::Mat output = _Output.getMat();
	
	/*std::cout << "Ǯ�� ���� �� input :\n" << zeroPaddingMat << std::endl;
	std::cout << "Ǯ�� ���� �� output :\n" << output << std::endl;
	std::cout << output.at<double>(0, 0) << std::endl;*/
	//���� �е� ��İ� Ŀ�η� ���� ��� ���� ��, ���� ����� output ��Ŀ� ����
	for (int y = 0; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {
			double maxValue = 0;
			for (int ky = 0; ky < poolSize.height; ky++) {
				for (int kx = 0; kx < poolSize.width; kx++) {
					if (maxValue < zeroPaddingMat.at<double>(y * stride.height + ky, x * stride.width + kx))
						maxValue = zeroPaddingMat.at<double>(y * stride.height + ky, x * stride.width + kx);
				}
			}
			//std::cout << maxValue << std::endl;
			output.at<double>(y, x) = maxValue;
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
					if (poolResult.at<double>(y,x) == poolInput.at<double>(y * stride.height + ky, x * stride.width + kx)) {
						poolFilter.at<double>(y * stride.height + ky, x * stride.width + kx) = 1;
						findFlag = true;
						break;
					}
				}
			}
		}
	}
}

void Math::MaxPoolingBackprop(cv::InputArray _Input, cv::OutputArray _Output, cv::InputArray _PoolFilter, const cv::Size& poolSize, const cv::Size& stride)
{
	cv::Mat zeroPaddingMat = _Input.getMat(); 
	cv::Mat poolFilterMat = _PoolFilter.getMat();

	//�Է� ����� ���� �е� ����� �ƴ� ��� �����е�
	if ((zeroPaddingMat.rows - poolSize.height) / stride.height + 1 != _Input.getMat().rows
		|| (zeroPaddingMat.cols - poolSize.width) / stride.width + 1 != _Input.getMat().cols) {
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, zeroPaddingMat.size(), poolSize, stride);
	}

	zeroPaddingMat.copyTo(_Output);
	cv::Mat outputMat = _Output.getMat();

	outputMat = outputMat.mul(poolFilterMat);
	outputMat.copyTo(_Output);
}


void Math::GetConvBackpropFilters(cv::InputArray _Input, std::vector<std::pair<int, int>>* _Output, cv::InputArray k, const cv::Size& stride)
{
	cv::Mat input = _Input.getMat();
	cv::Mat kernel = k.getMat();

	//���� �е� ��� ���� �ùķ��̼����� zeroPaddingMatrix�� ���� inputMatrix Offset�� ��´�
	cv::Rect inputOffset(0,0,0,0);

	double p = 0;
	int oH = (int)((input.rows + 2 * p - k.size().height) / stride.height) + 1;
	int oW = (int)((input.cols + 2 * p - k.size().width) / stride.width) + 1;


	//�е��� 0.5 �þ �����ٿ��� + ��, ������ + �Ʒ� ������ ��� Ȯ��
	while (oH != input.rows) {
		p += 0.5;
		if (p - (int)p != 0)
			inputOffset.y++;
			//cv::copyMakeBorder(_Output, _Output, 1, 0, 0, 0, cv::BORDER_CONSTANT, 0);
		else
			inputOffset.height++;
			//cv::copyMakeBorder(_Output, _Output, 0, 1, 0, 0, cv::BORDER_CONSTANT, 0);

		oH = (int)((input.rows + 2 * p - k.size().height) / stride.height) + 1;
	}
	p = 0;
	while (oW != input.cols) {
		p += 0.5;
		if (p - (int)p != 0)
			inputOffset.x++;
			//cv::copyMakeBorder(_Output, _Output, 0, 0, 1, 0, cv::BORDER_CONSTANT, 0);
		else
			inputOffset.width++;
			//cv::copyMakeBorder(_Output, _Output, 0, 0, 0, 1, cv::BORDER_CONSTANT, 0);

		oW = (int)((input.cols + 2 * p - k.size().width) / stride.width) + 1;
	}
	
	int stX, stY, edX, edY;

	//std::cout << "Ŀ�� ����" << input.size() << "ũ���� �Է� ��� �ռ��� ���� ��� : " << std::endl;
	//���� �е� �����̹Ƿ� input Matrix Size = output Matrix Size
	for (int outputY = 0; outputY < input.rows; outputY++) {
		for (int outputX = 0; outputX < input.cols; outputX++) {
			//start��ǥ
			stX = inputOffset.x - outputX * stride.width;
			stY = inputOffset.y - outputY * stride.height;
			edX = stX + (input.cols - 1);
			edY = stY + (input.rows - 1);
			if (edX > k.size().width - 1)
				edX = k.size().width - 1;
			if (edY > k.size().width - 1)
				edY = k.size().width - 1;
			if (stX < 0)
				stX = 0;
			if (stY < 0)
				stY = 0;

			//start��ǥ
			_Output->at((unsigned long long)outputY * input.cols + outputX).first = (stY)*k.size().width + (stX);
			//end��ǥ
			_Output->at((unsigned long long)outputY * input.cols + outputX).second = (edY)*k.size().width + (edX);

			//std::cout << stX << "," << stY << " "<< edX << "," << edY << "|";
		}
		//std::cout << std::endl;
	}
}


void Math::ConvKBackprop(cv::InputArray _Input, cv::InputArray _ConvZeroPadInput, cv::InputArray _Kernel, cv::OutputArray _Output, const std::vector<std::pair<int, int>>& _ConvFilter, const cv::Size& stride, double learningRate)
{
	cv::Mat zeroPaddingMat = _ConvZeroPadInput.getMat();
	const int filterRows = _Kernel.size().height;
	const int filterCols = _Kernel.size().width;
	//�ռ��� �Է� ����� ���� �е� ����� �ƴ� ��� �����е�
	if ((zeroPaddingMat.rows - filterRows) / stride.height + 1 != _Input.getMat().rows
		|| (zeroPaddingMat.cols - filterCols) / stride.width + 1 != _Input.getMat().cols) {
		//�ռ��� �Լ� ����� ���� ũ�Ⱑ �ǵ��� �����е� �߰�
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, _Input.getMat().size(), cv::Size(filterCols, filterRows), stride);
		std::cout << "ConvKBackprop ����ġ ���� �Է� : �ռ��� �Է��� �ڵ����� �����е��߽��ϴ�. :\n"<<zeroPaddingMat << std::endl;
	}
	cv::Mat input = _Input.getMat();
	cv::Mat kernel = _Kernel.getMat();
	
	//_Kernel.copyTo(_Output);
	_Output.create(_Kernel.size(), CV_64FC1);
	_Output.setTo(0);
	cv::Mat kOutput = _Output.getMat();

	//std::cout << "Ŀ�� ������Ʈ" << std::endl;
	//�Ҽ��� 4�ڸ����� ���
	//std::cout << std::fixed;
	//std::cout.precision(4);
	//input����� ũ��� �ռ����� ���� �е����� ����Ǳ⿡ �ռ��� ��� ��İ� ���� ũ��
	for (int iY = 0; iY < _Input.size().height; iY++) {
		for (int iX = 0; iX < _Input.size().width; iX++) {
			if (input.at<double>(iY, iX) == 0)
				continue;
#pragma region �ռ��� ���ͷ� Ŀ�� ��Ŀ� �����ϴ� Input ��� ��Ҹ� ����
			cv::Mat kTemp = cv::Mat(kOutput.size(), CV_64FC1);
			kTemp.setTo(0);

			int fIndex = iY * _Input.size().width + iX;

			int fYStart = (int)(_ConvFilter[fIndex].first / kernel.cols);
			int fXStart = _ConvFilter[fIndex].first % kernel.cols;
			int fYEnd = (int)(_ConvFilter[fIndex].second / kernel.cols);
			int fXEnd = _ConvFilter[fIndex].second % kernel.cols;
			for (int fY = fYStart; fY <= fYEnd; fY++) {
				for (int fX = fXStart; fX <= fXEnd; fX++) {
					//Kernel ������Ʈ
					kTemp.at<double>(fY, fX) += input.at<double>(iY, iX) * zeroPaddingMat.at<double>(iY + fY, iX + fX);
				}
			}
#pragma endregion
			//����� ����� ������ Ư�� ������ ǥ��ȭ
			kTemp /= (fYEnd - fYStart) * (fXEnd - fXStart);
			kOutput += kTemp;
			//std::cout << std::endl;
		}
	}
	kOutput *= learningRate;
	//std::cout << "Ŀ�� ������Ʈ ��� : \n"<<kOutput << std::endl;
	kOutput += _Kernel.getMat();
}

void Math::ConvXBackprop(cv::InputArray _Input, cv::InputArray _Kernel, cv::OutputArray _Output, const std::vector<std::pair<int, int>>& _ConvFilter, const cv::Size& stride, double learningRate)
{
	cv::Mat input = _Input.getMat();
	cv::Mat kernel = _Kernel.getMat();

	//_Kernel.copyTo(_Output);
	_Output.create(input.size(), CV_64FC1);
	_Output.setTo(0);
	cv::Mat output = _Output.getMat();

	//�Ҽ��� 4�ڸ����� ���
	//std::cout << std::fixed;
	//std::cout.precision(0);
	//input����� ũ��� �ռ����� ���� �е����� ����Ǳ⿡ �ռ��� ��� ��İ� ���� ũ��
	for (int iY = 0; iY < _Input.size().height; iY++) {
		for (int iX = 0; iX < _Input.size().width; iX++) {
			if (input.at<double>(iY, iX) == 0)
				continue;
#pragma region �ռ��� ���ͷ� Ŀ�� ��Ŀ� �����ϴ� Input ��� ��Ҹ� ����
			cv::Mat oTemp = cv::Mat(output.size(), CV_64FC1);
			oTemp.setTo(0);

			int fIndex = iY * _Input.size().width + iX;

			int fYStart = (int)(_ConvFilter[fIndex].first / kernel.cols);
			int fXStart = _ConvFilter[fIndex].first % kernel.cols;
			int fYEnd = (int)(_ConvFilter[fIndex].second / kernel.cols);
			int fXEnd = _ConvFilter[fIndex].second % kernel.cols;
			for (int fY = fYStart; fY <= fYEnd; fY++) {
				for (int fX = fXStart; fX <= fXEnd; fX++) {
					//std::cout << "K" << fY * kernel.cols + fX << "+=" << input.at<double>(iY, iX) * zeroPaddingMat.at<double>(iY + fY, iX + fX) << "|";
					output.at<double>(iY, iX) += input.at<double>(iY, iX) * kernel.at<double>(fY, fX);
				}
			}
#pragma endregion
			//����� ����� ������ Ư�� ������ ǥ��ȭ
			oTemp /= (fYEnd - fYStart) * (fXEnd - fXStart);
			output += oTemp;
			//std::cout << std::endl;
		}
	}
	output *= learningRate;
}

void Math::NeuralNetwork(cv::InputArray _Input, cv::OutputArray _Output, cv::InputArray w)
{
	cv::Mat output;
	_Input.getMat().copyTo(output);
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
	double max = 0;

	//softmax�� ���Ѵ�� �߻����� �ʰ� max�� ���� ����ó���Ѵ�.
	//https://leedakyeong.tistory.com/entry/�عٴں���-�����ϴ�-������-����Ʈ�ƽ�-�Լ�-�����ϱ�-in-���̽�-softmax-in-python
	for (int y = 0; y < input.rows; y++) {
		sum = 0;
		max = 0;
		
		//�ش� ���� max�� ���Ѵ�
		cv::minMaxLoc(input.row(y), 0, &max, 0, 0);
		for (int x = 0; x < input.cols; x++) {
			output.at<double>(y, x) = exp(input.at<double>(y, x) - max);
			sum += output.at<double>(y, x);
		}
		output.row(y) /= sum;
	}

}