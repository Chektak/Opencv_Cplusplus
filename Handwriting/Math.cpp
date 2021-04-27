#include "Math.h"

void Math::CreateZeroPadding(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& convResultSize, const cv::Size& k, const cv::Size& stride)
{
	cv::Mat input = _Input.getMat();
	_Input.copyTo(_Output);
	
	//패딩 맞추기 자동화(합성곱 출력 크기 계산 https://excelsior-cjh.tistory.com/79)
	double p = 0;
	int oH = (int)((input.rows + 2 * p - k.height) / stride.height) + 1;
	int oW = (int)((input.cols + 2 * p - k.width) / stride.width) + 1;

	//제로 패딩 행렬 생성
	//패딩이 0.5 늘어날 때마다 왼쪽 + 위, 오른쪽 + 아래 순으로 행렬 확장
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
					output.at<double>(y, x) += kernel.at<double>(ky, kx)
						* zeroPaddingMat.at<double>(y * stride.height + ky, x * stride.width + kx);
				}
			}
		}
	}
	//평균을 계산해 간단한 특성 스케일 표준화
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

	//제로 패딩 행렬이 유효하지 않은지 검사(풀링 성립 유무)
	double outputHeight = (double)(zeroPaddingMat.rows - poolSize.height) / stride.height + 1;
	double outputWidth = (double)(zeroPaddingMat.cols - poolSize.width) / stride.width + 1;
	//std::cout << "풀링 함수 안에서 input size :\n" << outputWidth << std::endl<< outputHeight << std::endl;

	if (outputHeight != (int)outputHeight
		|| outputWidth != (int)outputWidth) {
		//std::cout << "풀링 연산 불가하므로 제로 패딩 추가" << std::endl;
		outputHeight = cvRound(outputHeight);
		outputWidth = cvRound(outputWidth);
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, cv::Size((int)outputWidth, (int)outputHeight),poolSize, stride);
	}
	_Output.create(cv::Size((int)outputWidth, (int)outputHeight), zeroPaddingMat.type());
	_Output.setTo(0);
	cv::Mat output = _Output.getMat();
	
	/*std::cout << "풀링 연산 전 input :\n" << zeroPaddingMat << std::endl;
	std::cout << "풀링 연산 전 output :\n" << output << std::endl;
	std::cout << output.at<double>(0, 0) << std::endl;*/
	//제로 패딩 행렬과 커널로 교차 상관 연산 후, 연산 결과를 output 행렬에 저장
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
	
	//PoolFilter를 PoolInput과 같은 크기로 생성
	_PoolFilter.create(_PoolInput.size(), _PoolInput.type());
	_PoolFilter.setTo(0);
	cv::Mat poolFilter = _PoolFilter.getMat();
	
	//poolResult 행렬 요소에 대응하는 입력 행렬 요소를 찾아, 풀링 필터를 활성화
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

	//입력 행렬이 제로 패딩 행렬이 아닐 경우 제로패딩
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

	//제로 패딩 행렬 생성 시뮬레이션으로 zeroPaddingMatrix에 대한 inputMatrix Offset을 얻는다
	cv::Rect inputOffset(0,0,0,0);

	double p = 0;
	int oH = (int)((input.rows + 2 * p - k.size().height) / stride.height) + 1;
	int oW = (int)((input.cols + 2 * p - k.size().width) / stride.width) + 1;


	//패딩이 0.5 늘어날 때마다왼쪽 + 위, 오른쪽 + 아래 순으로 행렬 확장
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

	//std::cout << "커널 기준" << input.size() << "크기의 입력 행렬 합성곱 범위 출력 : " << std::endl;
	//세임 패딩 연산이므로 input Matrix Size = output Matrix Size
	for (int outputY = 0; outputY < input.rows; outputY++) {
		for (int outputX = 0; outputX < input.cols; outputX++) {
			//start좌표
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

			//start좌표
			_Output->at((unsigned long long)outputY * input.cols + outputX).first = (stY)*k.size().width + (stX);
			//end좌표
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
	//합성곱 입력 행렬이 제로 패딩 행렬이 아닐 경우 제로패딩
	if ((zeroPaddingMat.rows - filterRows) / stride.height + 1 != _Input.getMat().rows
		|| (zeroPaddingMat.cols - filterCols) / stride.width + 1 != _Input.getMat().cols) {
		//합성곱 함수 결과와 같은 크기가 되도록 제로패딩 추가
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, _Input.getMat().size(), cv::Size(filterCols, filterRows), stride);
		std::cout << "ConvKBackprop 예기치 않은 입력 : 합성곱 입력을 자동으로 제로패딩했습니다. :\n"<<zeroPaddingMat << std::endl;
	}
	cv::Mat input = _Input.getMat();
	cv::Mat kernel = _Kernel.getMat();
	
	//_Kernel.copyTo(_Output);
	_Output.create(_Kernel.size(), CV_64FC1);
	_Output.setTo(0);
	cv::Mat kOutput = _Output.getMat();

	//std::cout << "커널 업데이트" << std::endl;
	//소수점 4자리까지 출력
	//std::cout << std::fixed;
	//std::cout.precision(4);
	//input행렬의 크기는 합성곱이 세임 패딩으로 진행되기에 합성곱 결과 행렬과 같은 크기
	for (int iY = 0; iY < _Input.size().height; iY++) {
		for (int iX = 0; iX < _Input.size().width; iX++) {
			if (input.at<double>(iY, iX) == 0)
				continue;
#pragma region 합성곱 필터로 커널 행렬에 대응하는 Input 행렬 요소를 더함
			cv::Mat kTemp = cv::Mat(kOutput.size(), CV_64FC1);
			kTemp.setTo(0);

			int fIndex = iY * _Input.size().width + iX;

			int fYStart = (int)(_ConvFilter[fIndex].first / kernel.cols);
			int fXStart = _ConvFilter[fIndex].first % kernel.cols;
			int fYEnd = (int)(_ConvFilter[fIndex].second / kernel.cols);
			int fXEnd = _ConvFilter[fIndex].second % kernel.cols;
			for (int fY = fYStart; fY <= fYEnd; fY++) {
				for (int fX = fXStart; fX <= fXEnd; fX++) {
					//Kernel 업데이트
					kTemp.at<double>(fY, fX) += input.at<double>(iY, iX) * zeroPaddingMat.at<double>(iY + fY, iX + fX);
				}
			}
#pragma endregion
			//평균을 계산해 간단한 특성 스케일 표준화
			kTemp /= (fYEnd - fYStart) * (fXEnd - fXStart);
			kOutput += kTemp;
			//std::cout << std::endl;
		}
	}
	kOutput *= learningRate;
	//std::cout << "커널 업데이트 행렬 : \n"<<kOutput << std::endl;
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

	//소수점 4자리까지 출력
	//std::cout << std::fixed;
	//std::cout.precision(0);
	//input행렬의 크기는 합성곱이 세임 패딩으로 진행되기에 합성곱 결과 행렬과 같은 크기
	for (int iY = 0; iY < _Input.size().height; iY++) {
		for (int iX = 0; iX < _Input.size().width; iX++) {
			if (input.at<double>(iY, iX) == 0)
				continue;
#pragma region 합성곱 필터로 커널 행렬에 대응하는 Input 행렬 요소를 더함
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
			//평균을 계산해 간단한 특성 스케일 표준화
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

	//softmax가 무한대로 발산하지 않게 max를 구해 예외처리한다.
	//https://leedakyeong.tistory.com/entry/밑바닥부터-시작하는-딥러닝-소프트맥스-함수-구현하기-in-파이썬-softmax-in-python
	for (int y = 0; y < input.rows; y++) {
		sum = 0;
		max = 0;
		
		//해당 열의 max를 구한다
		cv::minMaxLoc(input.row(y), 0, &max, 0, 0);
		for (int x = 0; x < input.cols; x++) {
			output.at<double>(y, x) = exp(input.at<double>(y, x) - max);
			sum += output.at<double>(y, x);
		}
		output.row(y) /= sum;
	}

}