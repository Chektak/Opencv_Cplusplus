#include "CNNMachine.h"

void CNNMachine::Training(int epoch, double learningRate, double l2)
{
	this->learningRate = learningRate;
	
	lossAverages.push_back(0);
	//epoch가 0 이하로 입력되면 무한 반복
	for (nowEpoch = 1; (epoch <= 0) ? true : nowEpoch <= epoch; nowEpoch++) {
		std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
		std::cout << nowEpoch << "번째 훈련" << std::endl;
		std::cout << "[명령어 0 : Menu 열기 | 1 : Auto Training | 2 : Model Save 모델 저장 | 3 : Model Load 모델 불러오기" << std::endl;
		std::cout << "[ 4 : Hyper Parameter 하이퍼 파라미터 설정 | Enter : Debug Log 훈련 행렬 출력 ] " << std::endl;
		
		//autoTraining이 true면 delay마다 자동 진행, false면 입력까지 기다리기
		//키 버퍼 초기화
		int key = 0;
		//while (key != -1) {
		//	key = cv::waitKeyEx(1);
		//}
		key = cv::waitKeyEx((int)autoTraining * ((autoTrainingDelay <= 0) ? trainingTickMeter.getTimeMilli() + 200 : autoTrainingDelay));
		if (KeyEvent(key) == false){
			continue;
		}

		trainingTickMeter.reset();
		trainingTickMeter.start();

#pragma region 정방향, 모델 손실(코스트), 역방향 계산
		std::cout << "정방향 계산" << std::endl;
		ForwardPropagation();
		loss = 0;
		for (int y = 0; y < yMat.rows; y++) {
			for (int x = 0; x < yMat.cols; x++) {
				//log(0) 음의 무한대 예외처리로 0 대신 0에 가까운 수 사용
				loss += yMat.at<double>(y, x) * log((yHatMat.at<double>(y, x) == 0) ? 0.00000000001 : yHatMat.at<double>(y, x));
			}
		}
		//loss /= -yMat.rows;
		loss *= -1;
		lossAverages.push_back(lossAverages[nowEpoch - 1] + loss / nowEpoch);
		std::cout << "코스트 : " << loss << std::endl;
		std::cout << "코스트 평균 : " << lossAverages[nowEpoch] << std::endl;
		std::cout << "역방향 계산 중..." << std::endl;
		BackPropagation(learningRate);
#pragma endregion

		trainingTickMeter.stop();
	}
}

void CNNMachine::Init(OpencvPractice* op, int useData_Num, int kernel1_Num, int kernel2_Num, int classification_Num)
{
	
	std::vector<cv::Mat> trainingVec;	   
	std::vector<uchar> labelVec;
	this->useData_Num = useData_Num;
	this->kernel1_Num = kernel1_Num;
	this->kernel2_Num = kernel2_Num;
	this->classification_Num = classification_Num;

	op->MnistTrainingDataRead("Resources/train-images.idx3-ubyte", trainingVec, useData_Num);
	op->MnistLabelDataRead("Resources/train-labels.idx1-ubyte", labelVec, useData_Num);

	for (int i = 0; i < trainingVec.size(); i++) {
		trainingMats.push_back(cv::Mat());
		x1ZeroPaddingMats.push_back(cv::Mat_<double>());

		//uchar형 행렬 요소를 double형 행렬 요소로 타입 캐스팅
		trainingVec[i].convertTo(trainingMats[i], CV_64FC1);
		//평균을 계산해 간단한 특성 스케일 표준화
		trainingMats[i] /= 255;
	}

	

	poolStride = cv::Size(2, 2);
	poolSize = cv::Size(2, 2);

#pragma region 모든 가중치 행렬을 균등 분포로 랜덤 초기화, 커널 역방향 계산 필터 초기화
	cv::RNG gen(cv::getTickCount());
	//커널 1은 채널 한개(입력층 채널이 흑백 단일)
	
	kernels1.push_back(std::vector<cv::Mat>());
	for (int k1i = 0; k1i < kernel1_Num; k1i++) {
		kernels1[0].push_back(cv::Mat(cv::Size(3, 3), CV_64FC1));
		gen.fill(kernels1[0][k1i], cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(2));

		kernels2.push_back(std::vector<cv::Mat>());
		//커널 2는 채널이 커널 1의 개수
		for (int k2i = 0; k2i < kernel2_Num; k2i++) {
			kernels2[k1i].push_back(cv::Mat(cv::Size(3, 3), CV_64FC1));
			gen.fill(kernels2[k1i][k2i], cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(2));
		}
		
		
	}
	kernel1Stride = cv::Size(1, 1);
	kernel2Stride = cv::Size(1, 1);
	
	//합성곱은 세임 패딩으로 진행하므로 풀링층 2개에서의 축소만 계산
	int wHeight = (trainingMats[0].rows - poolSize.height) / poolStride.height + 1;
		wHeight = (wHeight - poolSize.height) / poolStride.height + 1;
	int wWidth = (trainingMats[0].cols - poolSize.width) / poolStride.width + 1;
		wWidth = (wWidth - poolSize.width) / poolStride.width + 1;
	w1Mat.create(cv::Size(10, wHeight * wWidth * kernel2_Num), CV_64FC1);
	gen.fill(w1Mat, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(1));
	w2Mat.create(cv::Size(classification_Num, w1Mat.cols), CV_64FC1);
	gen.fill(w2Mat, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(1));

	
#pragma endregion

#pragma region 합성곱 결과, 합성곱 결과 제로 패딩, 풀링 결과, 풀링 결과 제로 패딩, 풀링 역방향 계산 필터 행렬 초기화

	for (int i = 0; i < trainingMats.size(); i++) {
		conv1Mats.push_back(std::vector<cv::Mat>());
		conv2Mats.push_back(std::vector<cv::Mat>());
		conv1Bias.push_back(std::vector<double>());
		conv2Bias.push_back(std::vector<double>());
		conv1ZeroPaddingMats.push_back(std::vector<cv::Mat>());
		conv2ZeroPaddingMats.push_back(std::vector<cv::Mat>());
		pool1result.push_back(std::vector<cv::Mat>());
		pool2result.push_back(std::vector<cv::Mat>());
		pool1resultZeroPadding.push_back(std::vector<cv::Mat>());

		pool1BackpropFilters.push_back(std::vector<cv::Mat>());
		pool2BackpropFilters.push_back(std::vector<cv::Mat>());
		for (int j = 0; j < kernel1_Num; j++) {
			conv1Mats[i].push_back(cv::Mat_<double>());
			conv1Bias[i].push_back(0);
			conv1ZeroPaddingMats[i].push_back(cv::Mat_<double>());
			pool1result[i].push_back(cv::Mat_<double>());
			pool1resultZeroPadding[i].push_back(cv::Mat_<double>());
		
			pool1BackpropFilters[i].push_back(cv::Mat_<double>());
		}
		for (int j = 0; j < kernel2_Num; j++) {
			conv2Mats[i].push_back(cv::Mat_<double>());
			conv2Bias[i].push_back(0);
			conv2ZeroPaddingMats[i].push_back(cv::Mat_<double>());
			pool2result[i].push_back(cv::Mat_<double>());
			
			pool2BackpropFilters[i].push_back(cv::Mat_<double>());
		}
	}
#pragma endregion

#pragma region 합성곱 역방향 사용 변수 초기화
	//합성곱 시 세임 패딩만 사용하므로 풀링 결과 크기만 계산
	pool1ResultSize =
		cv::Size(
			(trainingMats[0].size().width - poolSize.width) / poolStride.width + 1,
			(trainingMats[0].size().height - poolSize.height) / poolStride.height + 1
		);
	pool2ResultSize =
		cv::Size(
			(pool1ResultSize.width - poolSize.width) / poolStride.width + 1,
			(pool1ResultSize.height - poolSize.height) / poolStride.height + 1
		);
	//1번째 합성곱의 역방향 계산 필터 초기화
	int r1Size = trainingMats[0].rows * trainingMats[0].cols;
	for (int r1i = 0; r1i < r1Size; r1i++) {
		conv1BackpropFilters.push_back(std::pair<int, int>());
	}
	//2번째 합성곱 커널의 역방향 계산 필터 초기화
	int r2Size = pool1ResultSize.width * pool1ResultSize.height;
	for (int r2i = 0; r2i < r2Size; r2i++) {
		conv2BackpropFilters.push_back(std::pair<int, int>());
	}
	
	//손실 함수를 합성곱2 결과에 대해 미분한 행렬 초기화
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		yLossW2Relu3W1UpRelu2.push_back(std::vector<cv::Mat>());
		yLossW2Relu3W1UpRelu2P1UpRelu.push_back(std::vector<cv::Mat>());
		for (int k2n = 0; k2n < kernel2_Num; k2n++) {
			yLossW2Relu3W1UpRelu2[x1i].push_back(cv::Mat());
		}
		for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
			yLossW2Relu3W1UpRelu2P1UpRelu[x1i].push_back(cv::Mat());
		}
	}
#pragma endregion

#pragma region 정답 데이터 행렬과 가설 행렬 초기화
	//정답 데이터를 벡터로 변환한다.
	yMat = cv::Mat::zeros(cv::Size(classification_Num, trainingMats.size()), CV_64FC1);
	for (int y = 0; y < labelVec.size(); y++) {
		//열과 맞는다면 true(1), 아니라면 false(0)를 저장
		yMat.at<double>(y, labelVec[y]) = 1;
	}

	yHatMat.create(cv::Size(classification_Num, trainingMats.size()), CV_64FC1);
#pragma endregion
}

void CNNMachine::ForwardPropagation()
{
	//합성곱층 결과 행렬을 완전연결신경망 입력으로 변환할 때 사용
	std::vector<std::vector<double>> tempArr;
	cv::Mat tempMat;

	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		tempArr.push_back(std::vector<double>());

		Math::CreateZeroPadding(trainingMats[x1i], x1ZeroPaddingMats[x1i], trainingMats[0].size(), kernels1[0][0].size(), kernel1Stride);
		//합성곱층 1
		for (int k1i = 0; k1i < kernel1_Num; k1i++) {
			Math::Convolution(x1ZeroPaddingMats[x1i], conv1Mats[x1i][k1i], trainingMats[0].size(), kernels1[0][k1i], kernel1Stride);
			conv1Mats[x1i][k1i] += conv1Bias[x1i][k1i];
			Math::Relu(conv1Mats[x1i][k1i], conv1Mats[x1i][k1i]);
			Math::CreateZeroPadding(conv1Mats[x1i][k1i], conv1ZeroPaddingMats[x1i][k1i], pool1ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv1ZeroPaddingMats[x1i][k1i], pool1result[x1i][k1i], poolSize, poolStride);

			Math::CreateZeroPadding(pool1result[x1i][k1i], pool1resultZeroPadding[x1i][k1i], pool1result[0][0].size(), kernels2[0][0].size(), kernel2Stride);
		}
		//합성곱층 2
		/*합성곱층 1의 (행:데이터 수, 열:채널 수)의 이미지을 가진 pool1result행렬과
		합성곱층 2의 kernel2행렬을 행렬곱하듯 연결*/
		for (int k2i = 0; k2i < kernel2_Num; k2i++) {
			for (int k1i = 0; k1i < kernel1_Num; k1i++) {
				Math::Convolution(pool1resultZeroPadding[x1i][k1i], conv2Mats[x1i][k2i], pool1result[0][0].size(), kernels2[k1i][k2i], kernel2Stride);
			}
			conv2Mats[x1i][k2i] += conv2Bias[x1i][k2i];
			Math::Relu(conv2Mats[x1i][k2i], conv2Mats[x1i][k2i]);
			Math::CreateZeroPadding(conv2Mats[x1i][k2i], conv2ZeroPaddingMats[x1i][k2i], pool2ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv2ZeroPaddingMats[x1i][k2i], pool2result[x1i][k2i], poolSize, poolStride);
		}
		//완전연결신경망 입력
		//4차원 pool2result를 2차원 행렬 xMat으로 변환
		//vec<vec<Mat>> to vec<vec<double>> 변환 : https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
		//vec<vec<double>> to Mat 변환 : https://stackoverflow.com/questions/18519647/opencv-convert-vector-of-vector-to-mat
		for (int k2i = 0; k2i < kernel2_Num; k2i++) {
			tempMat = pool2result[x1i][k2i];
			for (int i = 0; i < tempMat.rows; ++i) {
				tempArr[x1i].insert(tempArr[x1i].end(), tempMat.ptr<double>(i), tempMat.ptr<double>(i) + tempMat.cols * tempMat.channels());
			}
		}
	}

	xMat.create(cv::Size(0, tempArr[0].size()), CV_64FC1);

	//훈련 데이터 수만큼 반복
	for (int i = 0; i < tempArr.size(); ++i)
	{
		// Make a temporary cv::Mat row and add to NewSamples _without_ data copy
		cv::Mat Sample(1, tempArr[0].size(), CV_64FC1, tempArr[i].data());
		xMat.push_back(Sample);
	}
	Math::NeuralNetwork(xMat, a1Mat, w1Mat);
	a1Mat += bias1;
	Math::Relu(a1Mat, a1Mat);

	Math::NeuralNetwork(a1Mat, yHatMat, w2Mat);

	yHatMat += bias2;
	Math::SoftMax(yHatMat, yHatMat);
}

void CNNMachine::BackPropagation(double learningRate)
{
	yLoss = -(yMat - yHatMat); //손실함수를 SoftMax 함수 결과에 대해 미분한 값
	w2T = w2Mat.t();
	yLossW2 = yLoss*w2T; //손실 함수를 완전연결층2 입력(ReLu(aMat))에 대해 미분한 값
	yLossW2 /= yLoss.cols;//평균을 계산해 간단한 특성 스케일 표준화

	//Relu(a1Mat)과 벡터곱
	//(정방향 계산에서 이미 ReLU를 적용했으므로 생략)
	//Math::Relu(a1Mat, a1Mat);
	yLossW2Relu3 = yLossW2.mul(a1Mat); //손실함수를 완전연결층1 결과에 대해 미분한 값

	yLossW2Relu3W1 = yLossW2Relu3 * w1Mat.t();
	//평균을 계산해 간단한 특성 스케일 표준화
	yLossW2Relu3W1 /= yLossW2Relu3.cols;

	//yLossW2Relu3W1를 합성곱층 결과 크기로 차원 변환, 풀링2 필터로 Up-Sampleling, Relu(Conv2)행렬과 벡터곱
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		for (int k2n = 0; k2n < kernel2_Num; k2n++) {
			//Pooling 함수 역방향 계산으로 풀링 필터 할당
			Math::GetMaxPoolingFilter(conv2ZeroPaddingMats[x1i][k2n], pool2BackpropFilters[x1i][k2n], pool2result[x1i][k2n], poolSize, poolStride);
			//풀링 필터로 업샘플링
			cv::Mat sample = yLossW2Relu3W1.row(x1i).reshape(1, pool2result[0].size()).row(k2n).reshape(1, pool2result[0][0].rows);
			Math::MaxPoolingBackprop(sample, yLossW2Relu3W1UpRelu2[x1i][k2n], pool2BackpropFilters[x1i][k2n], poolSize, poolStride);

			//Relu 함수 역방향 계산
			//Up-Sampleling 결과 행렬과 Relu(Conv2)행렬을 벡터곱
			yLossW2Relu3W1UpRelu2[x1i][k2n] = yLossW2Relu3W1UpRelu2[x1i][k2n].mul(conv2Mats[x1i][k2n]);
		}
	}
	
	//커널2 역방향 계산을 위한 합성곱2 필터 계산
	Math::GetConvBackpropFilters(pool1result[0][0], &conv2BackpropFilters, kernels2[0][0], kernel2Stride);
	//커널1 역방향 계산을 위한 합성곱1 필터 계산
	Math::GetConvBackpropFilters(trainingMats[0], &conv1BackpropFilters, kernels1[0][0], kernel1Stride);

	//yLossW2Relu3W1UpRelu2행렬과 합성곱2 함수의 커널2에 대한 미분 행렬을 벡터곱하고, 풀링1 필터로 Up-Sampleling 후 Relu(Conv1)행렬과 벡터곱
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		//커널 1 개수만큼 반복
		for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
			cv::Mat yLossW2Relu3W1UpRelu2P1 = cv::Mat(yLossW2Relu3W1UpRelu2[x1i][0].size(), CV_64FC1);
			yLossW2Relu3W1UpRelu2P1.setTo(0);

			//커널 2 개수만큼 반복
			for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
				cv::Mat k2Temp;
				//yLossW2Relu3W1UpRelu2행렬과 합성곱2 함수의 커널2에 대한 미분 행렬을 벡터곱
				Math::ConvXBackprop(yLossW2Relu3W1UpRelu2[x1i][k2n], kernels2[k1n][k2n], k2Temp, conv2BackpropFilters, kernel1Stride, learningRate);
				yLossW2Relu3W1UpRelu2P1 += k2Temp;
			}
			//평균 계산으로 특성 스케일 표준화
			yLossW2Relu3W1UpRelu2P1 /= kernels2[0].size();

			//Pooling 함수 역방향 계산으로 풀링 필터 정의
			Math::GetMaxPoolingFilter(conv1ZeroPaddingMats[x1i][k1n], pool1BackpropFilters[x1i][k1n], pool1result[x1i][k1n], poolSize, poolStride);
			//풀링 필터로 업샘플링
			Math::MaxPoolingBackprop(yLossW2Relu3W1UpRelu2P1, yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n], pool1BackpropFilters[x1i][k1n], poolSize, poolStride);

			//Relu 함수 역방향 계산
			//(정방향 계산에서 이미 ReLU를 적용했으므로 생략)
			//Math::Relu(conv1Mats[x1i][k1n], conv1Mats[x1i][k1n]);
			yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n] = yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n].mul(conv1Mats[x1i][k1n]);
		}
	}
#pragma region 합성곱층1 가중치 행렬(커널1), 편향 역방향 계산
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
			for (int k1c = 0; k1c < kernels1.size(); k1c++) {
				cv::Mat newKernel;
				Math::ConvKBackprop(-yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n], x1ZeroPaddingMats[x1i], kernels1[k1c][k1n], newKernel, conv1BackpropFilters, kernel1Stride, learningRate);
				newKernel.copyTo(kernels1[k1c][k1n]);
			}
			double conv1BiasBackprop = 0;
			for (int bY = 0; bY < conv1Mats[0][0].rows; bY++) {
				for (int bX = 0; bX < conv1Mats[0][0].cols; bX++) {
					conv1BiasBackprop += yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n].at<double>(bY, bX);
				}
			}
			conv1Bias[x1i][k1n] -= learningRate * conv1BiasBackprop / conv1Mats[0][0].rows * conv1Mats[0][0].cols;
		}
	}
	
#pragma endregion

#pragma region 합성곱층2 가중치 행렬(커널2) 역방향 계산
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
			for (int k2c = 0; k2c < kernels2.size(); k2c++) {
				cv::Mat newKernel;
				Math::ConvKBackprop(-yLossW2Relu3W1UpRelu2[x1i][k2n], pool1resultZeroPadding[x1i][k2c], kernels2[k2c][k2n], newKernel, conv2BackpropFilters, kernel2Stride, learningRate);
				newKernel.copyTo(kernels2[k2c][k2n]);
			}
			double conv2BiasBackprop = 0;
			for (int bY = 0; bY < conv2Mats[0][0].rows; bY++) {
				for (int bX = 0; bX < conv2Mats[0][0].cols; bX++) {
					conv2BiasBackprop += yLossW2Relu3W1UpRelu2[x1i][k2n].at<double>(bY, bX);
				}
			}
			conv2Bias[x1i][k2n] -= learningRate * conv2BiasBackprop / conv2Mats[0][0].rows * conv2Mats[0][0].cols;
		}
	}

#pragma endregion

#pragma region 완전연결신경망층 가중치 행렬 역방향 계산
	w1Mat -= learningRate * xMat.t() * (yLossW2Relu3);// / yLossW2Relu3.rows;//평균을 계산해 간단한 특성 스케일 표준화
	double bias1Backprop = 0;
	for (int y = 0; y < yLossW2Relu3.rows; y++) {
		for (int x = 0; x < yLossW2Relu3.cols; x++) {
			bias1Backprop += yLossW2Relu3.at<double>(y, x);
		}
	}
	bias1 -= learningRate * bias1Backprop;// / yLossW2Relu3.rows * yLossW2Relu3.cols;
	
	w2Mat -= learningRate * (a1Mat.t() * (yLoss)) / yLoss.rows;//평균을 계산해 간단한 특성 스케일 표준화
	double bias2Backprop = 0;
	for (int y = 0; y < yLoss.rows; y++) {
		for (int x = 0; x < yLoss.cols; x++) {
			bias2Backprop += yLoss.at<double>(y, x);
		}
	}
	bias2 -= learningRate * bias2Backprop / yLoss.rows * yLoss.cols;
#pragma endregion
}

void CNNMachine::ModelPredict(cv::InputArray _Input)
{
	//합성곱층 결과 행렬을 완전연결신경망 입력으로 변환할 때 사용
	std::vector<double> tempArr;
	cv::Mat tempMat;

	cv::Mat input = _Input.getMat();
	cv::Mat inputZeroPad;

	tempArr.push_back(0);

	Math::CreateZeroPadding(input, inputZeroPad, input.size(), kernels1[0][0].size(), kernel1Stride);

	//합성곱층 1
	for (int k1i = 0; k1i < kernel1_Num; k1i++) {
		Math::Convolution(inputZeroPad, predictConv1Mats[k1i], trainingMats[0].size(), kernels1[0][k1i], kernel1Stride);
		predictConv1Mats[k1i] += conv1Bias[x1i][k1i];
		Math::Relu(predictConv1Mats[k1i], predictConv1Mats[k1i]);
		Math::CreateZeroPadding(predictConv1Mats[k1i], predictConv1ZeroPaddingMats[k1i], pool1ResultSize, poolSize, poolStride);
		Math::MaxPooling(predictConv1ZeroPaddingMats[k1i], predictPool1result[k1i], poolSize, poolStride);

		Math::CreateZeroPadding(predictPool1result[k1i], predictPool1resultZeroPadding[x1i][k1i], pool1result[0][0].size(), kernels2[0][0].size(), kernel2Stride);
	}
	//합성곱층 2
	/*합성곱층 1의 (행:데이터 수, 열:채널 수)의 이미지을 가진 pool1result행렬과
	합성곱층 2의 kernel2행렬을 행렬곱하듯 연결*/
	for (int k2i = 0; k2i < kernel2_Num; k2i++) {
		for (int k1i = 0; k1i < kernel1_Num; k1i++) {
			Math::Convolution(predictPool1resultZeroPadding[k1i], predictConv2Mats[k2i], pool1result[0][0].size(), kernels2[k1i][k2i], kernel2Stride);
		}
		predictConv2Mats[k2i] += conv2Bias[x1i][k2i];
		Math::Relu(predictConv2Mats[k2i], predictConv2Mats[k2i]);
		Math::CreateZeroPadding(predictConv2Mats[k2i], predictConv2ZeroPaddingMats[k2i], pool2ResultSize, poolSize, poolStride);
		Math::MaxPooling(predictConv2ZeroPaddingMats[k2i], predictPool2result[k2i], poolSize, poolStride);
	}
	//완전연결신경망 입력
	//4차원 pool2result를 2차원 행렬 xMat으로 변환
	//vec<vec<Mat>> to vec<vec<double>> 변환 : https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
	//vec<vec<double>> to Mat 변환 : https://stackoverflow.com/questions/18519647/opencv-convert-vector-of-vector-to-mat
	for (int k2i = 0; k2i < kernel2_Num; k2i++) {
		tempMat = predictPool2result[k2i];
		for (int i = 0; i < tempMat.rows; ++i) {
			tempArr.insert(tempArr.end(), tempMat.ptr<double>(i), tempMat.ptr<double>(i) + tempMat.cols * tempMat.channels());
		}
	}

	cv::Mat predictNeuralInputMat;
	predictNeuralInputMat.create(cv::Size(0, tempArr.size()), CV_64FC1);

	cv::Mat Sample(1, tempArr.size(), CV_64FC1, tempArr.data());
	predictXMat.push_back(Sample);

	Math::NeuralNetwork(predictXMat, predictA1Mat, w1Mat);
	predictA1Mat += bias1;
	Math::Relu(predictA1Mat, predictA1Mat);

	Math::NeuralNetwork(predictA1Mat, predictYHatMat, w2Mat);

	predictYHatMat += bias2;
	Math::SoftMax(predictYHatMat, predictYHatMat);
}

bool CNNMachine::SaveModel(cv::String fileName)
{
	cv::FileStorage fs = cv::FileStorage(fileName, cv::FileStorage::WRITE);

	if (!fs.isOpened()) {
		std::cerr << "File open Failed!!!" << std::endl;
		return false;
	}
	fs << "LearningRate"		<< learningRate;
	fs << "loss"				<< loss;
	fs << "lossAverages"		<< lossAverages;
	fs << "NowEpoch"			<< nowEpoch;
	fs << "USEDATA_NUM"			<< useData_Num;
	fs << "KERNEL1_NUM"			<< kernel1_Num;
	fs << "KERNEL2_NUM"			<< kernel2_Num;
	fs << "CLASSIFICATION_NUM"	<< classification_Num;
	fs << "L1"					<< 0;
	fs << "L2"					<< 0;
	fs << "PoolSize"			<< poolSize;
	fs << "PoolStride"			<< poolStride;
	fs << "Kernel1Stride"		<< kernel1Stride;
	fs << "Kernel2Stride"		<< kernel2Stride;

	fs << "Kernels1"			<< kernels1;
	fs << "Kernels2"			<< kernels2;
	fs << "Conv1Bias"			<< conv1Bias;
	fs << "Conv2Bias"			<< conv2Bias;
	fs << "w1Mat"				<< w1Mat;
	fs << "w2Mat"				<< w2Mat;
	fs << "Bias1"				<< bias1;
	fs << "Bias2"				<< bias2;

	std::cout << "Model Save Completed!!!" << std::endl;
	fs.release();
	return true;
}

bool CNNMachine::LoadModel(cv::String fileName)
{
	cv::FileStorage fs = cv::FileStorage(fileName, cv::FileStorage::READ);

	if (!fs.isOpened()) {
		std::cerr << "File open Failed!!!" << std::endl;
		return false;
	}
	std::cout << kernels1.size() << std::endl;

	fs["LearningRate"]		>> learningRate;
	fs["loss"]				>> loss;
	fs["NowEpoch"]			>> nowEpoch;
	fs["USEDATA_NUM"]		>> useData_Num;
	fs["KERNEL1_NUM"]		>> kernel1_Num;
	fs["KERNEL2_NUM"]		>> kernel2_Num;
	fs["CLASSIFICATION_NUM"]>> classification_Num;
	fs["L1"]				>> 0;
	fs["L2"]				>> 0;
	fs["PoolSize"]			>> poolSize;
	fs["PoolStride"]		>> poolStride;
	fs["Kernel1Stride"]		>> kernel1Stride;
	fs["Kernel2Stride"]		>> kernel2Stride;

	ReleaseVectors();
	Init(op, useData_Num, kernel1_Num, kernel2_Num, classification_Num);
	fs["lossAverages"]		>> lossAverages;
	fs["Kernels1"]			>> kernels1;
	fs["Kernels2"]			>> kernels2;
	fs["Conv1Bias"]			>> conv1Bias;
	fs["Conv2Bias"]			>> conv2Bias;
	fs["w1Mat"]				>> w1Mat;
	fs["w2Mat"]				>> w2Mat;
	fs["Bias1"]				>> bias1;
	fs["Bias2"]				>> bias2;
	std::cout << kernels1.size() << std::endl;

	std::cout << "Model Load Completed!!!" << std::endl;
	fs.release();
	return true;
}

void CNNMachine::ReleaseVectors()
{
	lossAverages.clear();

	trainingMats.clear();
	x1ZeroPaddingMats.clear(); 
	kernels1.clear();
	kernels2.clear(); 
	conv1Mats.clear();
	conv2Mats.clear();
	conv1Bias.clear();
	conv2Bias.clear();
	conv1ZeroPaddingMats.clear();
	conv2ZeroPaddingMats.clear();
	pool1result.clear();
	pool2result.clear();
	pool1resultZeroPadding.clear();
	pool1BackpropFilters.clear();
	pool2BackpropFilters.clear(); 
	conv2BackpropFilters.clear();
	conv1BackpropFilters.clear();
	yLossW2Relu3W1UpRelu2.clear();
	yLossW2Relu3W1UpRelu2P1UpRelu.clear();

	w1Mat.release();
	a1Mat.release();
}

bool CNNMachine::KeyEvent(int key)
{
	//입력 버퍼 초기화
	std::cin.clear();

	switch (key) {
	case 13: //enter키를 누르면 훈련 행렬 출력
	{
		std::cout << "정방향 계산에서 얻은 yMat, yHatMat, yLoss로 역방향 계산 끝. " << std::endl;
		std::cout << "yMat(정답 행렬) : " << std::endl;
		std::cout << yMat << std::endl;
		std::cout << "yHatMat(가설 행렬) : " << std::endl;
		std::cout << yHatMat << std::endl;
		std::cout << "yLoss (= -(yMat - yHatMat)) : " << std::endl;
		std::cout << yLoss << std::endl;
		std::cout << "kernels1[0][0] : " << std::endl;
		std::cout << kernels1[0][0] << std::endl;
		std::cout << "kernels2[0][0] : " << std::endl;
		std::cout << kernels2[0][0] << std::endl;
		std::cout << "conv1Bias[0][0]" << std::endl;
		std::cout << conv1Bias[0][0] << std::endl;
		std::cout << "conv1Bias[0][1]" << std::endl;
		std::cout << conv1Bias[0][1] << std::endl;

		std::cout << "conv2Bias[0][0]" << std::endl;
		std::cout << conv2Bias[0][0] << std::endl;
		std::cout << "conv2Bias[0][1]" << std::endl;
		std::cout << conv2Bias[0][1] << std::endl;
		std::cout << "w1Mat[0][0]" << std::endl;
		std::cout << w1Mat.at<double>(0, 0) << std::endl;
		std::cout << "w2Mat[0][0]" << std::endl;
		std::cout << w2Mat.at<double>(0, 0) << std::endl;
		nowEpoch--;
		return false;
		break;
	}
	case 48: { //0번 키를 누르면 메뉴 열기
		std::cout << "1 : AutoTraining Delay | 2 : Model Save/Load Name(JSON Format) | 3 : HandWriting" << std::endl;
		int temp = 0;
		std::cin >> temp;
		switch (temp) {
		case 1:
		{
			std::cout << "last Training operation time (ms): " << trainingTickMeter.getTimeMilli() << std::endl;
			std::cout << "새로운 AutoTraining Deley (ms) 입력(현재 Delay : " << ((autoTrainingDelay <= 0) ? trainingTickMeter.getTimeMilli() + 200 : autoTrainingDelay) << ")" << std::endl;
			std::cin >> temp;
			autoTrainingDelay = temp;
			std::cout << "Delay 가" << autoTrainingDelay << "으로 설정되었습니다." << std::endl;
			break;
		}case 2:
		{
			break;
		}case 3:
		{
			std::cout << "Enter : 손글씨 예측 종료 | Mouse LeftButton : Draw | Mouse RightButton : Erase" << std::endl;
			cv::Mat paintScreen;
			paintScreen.zeros(trainingMats[0].size(), CV_64FC1);

			op->PaintWindow(paintScreen, paintScreen, "Paint", paintScreen.size() * 20, 13, this);

			break;
		}
		}
		nowEpoch--;
		return false;
		break;
	}
	case 49: //1번 키를 누르면 자동 훈련
	{
		autoTraining = !autoTraining;
		if (autoTraining) {
			std::cout << "이제부터 자동 훈련을 진행합니다." << std::endl;
		}
		else {
			std::cout << "자동 훈련을 중지했습니다." << std::endl;
		}
		nowEpoch--;
		return false;
		break;
	}
	case 50: //2번 키를 누르면 모델 저장
	{
		//훈련을 한번이라도 한 경우에만 저장 가능
		if (nowEpoch > 1) {
			if (SaveModel("Model.json")) {
				std::cout << nowEpoch << "번째 훈련 모델로 저장에 성공" << std::endl;
			}
		}
		else
		{
			std::cout << "먼저 훈련을 1회 이상 실행하세요." << std::endl;
		}
		nowEpoch--;
		return false;
		break;
	}
	case 51: //3번 키를 누르면 모델 로드
	{
		if (LoadModel("Model.json")) {
			std::cout << "로드된 모델로 훈련 시작" << std::endl;
			nowEpoch--;
			return false;
		}
		nowEpoch--;
		return false;
		break;
	}
	case 52: //4번 키를 누르면 하이퍼 파라미터 설정
	{
		std::cout << "1 : Learning Rate 학습률 | 2 : L1 라쏘 | 3 : L2 릿지" << std::endl;
		int temp = 0;
		std::cin >> temp;
		switch (temp) {
		case 1:
		{
			std::cout << "새로운 Learning Rate 입력(현재 Learning Rate : " << learningRate << ")" << std::endl;
			double newRate = 0;
			std::cin >> newRate;
			learningRate = newRate;
			std::cout << "Learning Rate가 " << learningRate << "으로 설정되었습니다." << std::endl;
			break;
		}case 2:
		{
			break;
		}case 3:
		{
			break;
		}
		}
		nowEpoch--;
		return false;
		break;
	}
	default: {
		if (key == -1 && autoTraining) {
			//자동 훈련 시 코스트가 0.5 이하일 경우 자동 종료
			/*if (loss < 0.5) {
				if (SaveModel("Model.json")) {
					std::cout << "자동 훈련 : 코스트가 0.5 이하이므로 종료합니다" << std::endl;
				}
				else {
					std::cout << "저장 실패, 자동 훈련을 종료합니다" << std::endl;
					autoTraining = false;
				}
				nowEpoch--;
				return false;
			}*/
		}
	}
	}
	return true;
}
