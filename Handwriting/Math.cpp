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

	//제로 패딩 행렬이 유효하지 않은지 검사(풀링 성립 유무)
	float outputHeight = (float)(zeroPaddingMat.rows - poolSize.height) / stride.height + 1;
	float outputWidth = (float)(zeroPaddingMat.cols - poolSize.width) / stride.width + 1;
	//std::cout << "풀링 함수 안에서 input size :\n" << outputWidth << std::endl<< outputHeight << std::endl;

	if (outputHeight != (int)outputHeight
		|| outputWidth != (int)outputWidth) {
		//std::cout << "풀링 연산 불가하므로 제로 패딩 추가" << std::endl;
		outputHeight = cvRound(outputHeight);
		outputWidth = cvRound(outputWidth);
		Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, cv::Size(outputWidth, outputHeight),poolSize, stride);
	}
	_Output.create(cv::Size(outputWidth, outputHeight), zeroPaddingMat.type());
	_Output.setTo(0);
	cv::Mat output = _Output.getMat();
	
	/*std::cout << "풀링 연산 전 input :\n" << zeroPaddingMat << std::endl;
	std::cout << "풀링 연산 전 output :\n" << output << std::endl;
	std::cout << output.at<float>(0, 0) << std::endl;*/
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
			//std::cout << maxValue << std::endl;
			output.at<float>(y, x) = maxValue;
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
	cv::Mat zeroPaddingMat = _Input.getMat();
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
	
	//std::cout << "합성곱 출력 행렬" << std::endl;
	int stX, stY, edX, edY;

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

			//std::cout << outputX << ", " << outputY << " 좌표의 입력 행렬 합성곱 범위 : " << std::endl;
			//std::cout << "시작점 : " << stX << ", " << stY << std::endl;
			//std::cout << "종료점 : " << edX << ", " << edY << std::endl;
		}
	}
}


void Math::ConvKBackprop(cv::InputArray _Input, cv::OutputArray _Kernel, const std::vector<std::pair<int, int>>& _ConvFilter, const cv::Size& stride)
{
	//입력 행렬에 대응하는 필터 요소를 알기 위해
	//합성곱 함수 입력과 같은 크기가 되도록 Input행렬에 제로패딩 추가
	//cv::Mat zeroPaddingMat = _Input.getMat();
	//const int filterRows = _ConvFilter.size();
	//const int filterCols = _ConvFilter.at(0).size();
	////입력 행렬이 제로 패딩 행렬이 아닐 경우 제로패딩
	//if ((zeroPaddingMat.rows - filterRows) / stride.height + 1 != _Input.getMat().rows
	//	|| (zeroPaddingMat.cols - filterCols) / stride.width + 1 != _Input.getMat().cols) {
	//	Math::CreateZeroPadding(zeroPaddingMat, zeroPaddingMat, zeroPaddingMat.size(), cv::Size(filterCols, filterRows), stride);
	//}

	////커널 크기와 같은 Output 행렬 생성
	//_Kernel.create(cv::Size(filterCols, filterRows), CV_32FC1);
	//_Kernel.setTo(0);
	//cv::Mat output = _Kernel.getMat();
	cv::Mat input = _Input.getMat();
	cv::Mat kOutput = _Kernel.getMat();
	cv::Mat kernel;
	_Kernel.copyTo(kernel);

	for (int iY = 0; iY < input.rows; iY++) {
		for (int iX = 0; iX < input.cols; iX++) {
			//필터에서 얻은 입력 범위 정보만큼 반복
			int fIndex = iY * input.cols + iX;
			for (int f = _ConvFilter[fIndex].first; f < _ConvFilter[fIndex].second; f++) {
				//Kernel 업데이트
				//std::cout << (int)f / kernel.cols << "," << f % kernel.cols << "요소" << std::endl;
				
				kOutput.at<float>((int)f / kernel.cols, f % kernel.cols) += kernel.at<float>((int)f / kernel.cols, f % kernel.cols) * input.at<float>(iY, iX);
			}
		}
	}
}

void Math::ConvXBackprop(cv::InputArray _Input, cv::InputArray _Kernel, cv::OutputArray _Output, const std::vector<std::pair<int, int>>& _ConvFilter, const cv::Size& stride)
{
	cv::Mat input = _Input.getMat();
	cv::Mat output = cv::Mat(input.size(), CV_32FC1);
	output.setTo(0);
	
	cv::Mat kernel = _Kernel.getMat();

	for (int iY = 0; iY < input.rows; iY++) {
		for (int iX = 0; iX < input.cols; iX++) {
			//필터에서 얻은 입력 범위 정보만큼 반복
			int fIndex = iY * input.cols + iX;
			for (int f = _ConvFilter[fIndex].first; f < _ConvFilter[fIndex].second; f++) {
				//Kernel 업데이트
				//std::cout << (int)f / kernel.cols << "," << f % kernel.cols << "요소" << std::endl;

				output.at<float>(iY, iX) += kernel.at<float>((int)f / kernel.cols, f % kernel.cols) * input.at<float>(iY, iX);
			}
		}
	}
	output = output.mul(input);
	output.copyTo(_Output);
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
		sum = 0;
		max = 0;
		//열에 대한 max를 구한다
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