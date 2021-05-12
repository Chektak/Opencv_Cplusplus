#include "CNNMachine.h"

void CNNMachine::Training(int epoch, double learningRate, double l2, GD gradientDescent)
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

#pragma region 정방향, 역방향 계산
		std::cout << "정방향, 역방향 계산 중..." << std::endl;
		switch (gradientDescent)
		{
		case CNNMachine::STOCHASTIC:
		{
		//	//훈련 샘플 순서를 섞어 역방향 업데이트
		//	//c++ 셔플 함수 https://en.cppreference.com/w/cpp/algorithm/random_shuffle
		//	int min = 0;
		//	int max = trainingMats.size() - 1;
		//	std::vector<int> trainingIndexs;
		//	for (int i = min; i <= max; i++) {
		//		trainingIndexs.push_back(i);
		//	}

		//	std::random_device rd;
		//	std::mt19937 g(rd());

		//	std::shuffle(trainingIndexs.begin(), trainingIndexs.end(), g);

		//	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		//		ForwardPropagationStochastic(trainingIndexs[x1i]);
		//		BackPropagationStochastic(trainingIndexs[x1i]);
		//	}
		}
			break;
		case CNNMachine::MINI_BATCH:
		/*{
			for (int bi = 0; bi < 5; bi++)
				ForwardPropagationMiniBatch(bi);
		}*/
			break;
		case CNNMachine::BATCH:
		{
			ForwardPropagationBatch();
			BackPropagationBatch();
		}
			break;
		default:
		{
			ForwardPropagationBatch();
			BackPropagationBatch();
		}
			break;
		}

#pragma endregion
#pragma region 모델 손실(cost) 계산
		loss = 0;
		//훈련 세트 손실
		//for (int y = 0; y < trainingYMat.rows; y++) {
		//	for (int x = 0; x < trainingYMat.cols; x++) {
		//		//log(0) 음의 무한대 예외처리로 0 대신 0에 가까운 수 사용
		//		loss += trainingYMat.at<double>(y, x) * log((trainingNeuralYHatMat.at<double>(y, x) == 0) ? 0.00000000001 : trainingNeuralYHatMat.at<double>(y, x));
		//	}
		//}
		//loss /= -trainingYMat.rows;

		//검증 세트 손실
		//cv::Mat neuralYHatMat = cv::Mat(1, predictPool2Result[0].rows * predictPool2Result[0].cols * predictPool2Result[0].channels(), CV_64FC1);;
		//for (int y = 0; y < validationYMat.rows; y++) {
		//	ModelPredict(validationMats[y], neuralYHatMat);
		//	for (int x = 0; x < validationYMat.cols; x++) {
		//		//log(0) 음의 무한대 예외처리로 0 대신 0에 가까운 수 사용
		//		loss += validationYMat.at<double>(y, x) * log((neuralYHatMat.at<double>(0, x) == 0) ? 0.00000000001 : neuralYHatMat.at<double>(0, x));
		//	}
		//}
		for (int y = 0; y < validationYMat.rows; y++) {
			ModelPredict(validationMats[y], validationNeuralYHatMat);
			for (int x = 0; x < validationYMat.cols; x++) {
				//log(0) 음의 무한대 예외처리로 0 대신 0에 가까운 수 사용
				loss += validationYMat.at<double>(y, x) * log((validationNeuralYHatMat.at<double>(0, x) == 0) ? 0.00000000001 : validationNeuralYHatMat.at<double>(0, x));
			}
		}
		loss /= -validationYMat.rows;

		//테스트 세트 손실
		//for (int y = 0; y < trainingYMat.rows; y++) {
		//	for (int x = 0; x < trainingYMat.cols; x++) {
		//		//log(0) 음의 무한대 예외처리로 0 대신 0에 가까운 수 사용
		//		loss += trainingYMat.at<double>(y, x) * log((trainingNeuralYHatMat.at<double>(y, x) == 0) ? 0.00000000001 : trainingNeuralYHatMat.at<double>(y, x));
		//	}
		//}
		//loss /= -trainingYMat.rows;

		lossAverages.push_back(lossAverages[nowEpoch - 1] + loss / nowEpoch);
		std::cout << "코스트 : " << loss << std::endl;
		std::cout << "코스트 평균 : " << lossAverages[nowEpoch] << std::endl;
#pragma endregion
		trainingTickMeter.stop();
	}
}

void CNNMachine::SplitData(const std::vector<cv::Mat>& mnistImageMats, const std::vector<uint8_t>& mnistImageLabels, const int &trainingSetRatio, const int &validationSetRatio, const int &testSetRatio)
{
 	const double ratioAtom = mnistImageMats.size() / ((double)trainingSetRatio + validationSetRatio + testSetRatio);
	const int trainingSetSize = cvRound(ratioAtom * trainingSetRatio);
	const int validationSetSize = cvRound(ratioAtom * validationSetRatio);
	const int testSetSize = cvRound(ratioAtom * testSetRatio);
	std::cout << "데이터셋 분할 최소 단위 : " << ratioAtom << std::endl;
	std::cout << "데이터셋 분할 비율 : " << trainingSetRatio << "+" << validationSetRatio << "+" << testSetRatio << "+" << std::endl;
	std::cout << "데이터셋 분할 후 크기 계산 결과 : \n 전체 데이터 셋 크기 : " << mnistImageMats.size() << std::endl;
	std::cout << "\t" << trainingSetSize << "+" << validationSetSize << "+" << testSetSize << "+" << std::endl;
	for (int i = 0; i < trainingSetSize; i++) {
		trainingMats.push_back(cv::Mat());
		trainingConv1x1ZeroPaddingMats.push_back(cv::Mat_<double>());

		//uint8_t형 행렬 요소를 double형 행렬 요소로 타입 캐스팅
		mnistImageMats[i].convertTo(trainingMats[i], CV_64FC1);
		//평균을 계산해 간단한 특성 스케일 정규화
		trainingMats[i] /= 255;
	}
	for (int i = trainingSetSize; i < trainingSetSize + validationSetSize; i++) {
		validationMats.push_back(cv::Mat());

		//uint8_t형 행렬 요소를 double형 행렬 요소로 타입 캐스팅
		mnistImageMats[i].convertTo(validationMats[i-trainingSetSize], CV_64FC1);
		//평균을 계산해 간단한 특성 스케일 정규화
		validationMats[i - trainingSetSize] /= 255;
	}
	for (int i = trainingSetSize + validationSetSize; i < trainingSetSize + validationSetSize + testSetSize; i++) {
		testMats.push_back(cv::Mat());

		//uint8_t형 행렬 요소를 double형 행렬 요소로 타입 캐스팅
		mnistImageMats[i].convertTo(testMats[i - (trainingSetSize + validationSetSize)], CV_64FC1);
		//평균을 계산해 간단한 특성 스케일 정규화
		testMats[i - (trainingSetSize + validationSetSize)] /= 255;
	}

#pragma region 정답 데이터 행렬과 가설 행렬 초기화
	//정답 데이터를 벡터로 변환한다.
	trainingYMat = cv::Mat::zeros(cv::Size(classification_Num, trainingSetSize), CV_64FC1);
	for (int y = 0; y < trainingSetSize; y++) {
		//열과 맞는다면 true(1), 아니라면 false(0)를 저장
		trainingYMat.at<double>(y, mnistImageLabels[y]) = 1;
	}
	trainingNeuralYHatMat.create(cv::Size(classification_Num, trainingSetSize), CV_64FC1);
	validationYMat = cv::Mat::zeros(cv::Size(classification_Num, validationSetSize), CV_64FC1);
	std::cout << validationYMat.size << std::endl;
	for (int y = trainingSetSize; y < trainingSetSize + validationSetSize; y++) {
		//열과 맞는다면 true(1), 아니라면 false(0)를 저장
		validationYMat.at<double>(y - trainingSetSize, (int)mnistImageLabels[y]) = 1;
	}
	validationNeuralYHatMat.create(cv::Size(classification_Num, validationSetSize), CV_64FC1);
	testYMat = cv::Mat::zeros(cv::Size(classification_Num, testSetSize), CV_64FC1);
	for (int y = validationSetSize + trainingSetSize; y < trainingSetSize + validationSetSize + testSetSize; y++) {
		//열과 맞는다면 true(1), 아니라면 false(0)를 저장
		testYMat.at<double>(y - (validationSetSize + trainingSetSize), mnistImageLabels[y]) = 1;
	}
	testNeuralYHatMat.create(cv::Size(classification_Num, testSetSize), CV_64FC1);
#pragma endregion
}

void CNNMachine::Init(OpencvPractice* op, int useData_Num, int kernel1_Num, int kernel2_Num, int classification_Num)
{
	
	std::vector<cv::Mat> imageMats;	   
	std::vector<uint8_t> imageLabels;
	this->useData_Num = useData_Num;
	this->kernel1_Num = kernel1_Num;
	this->kernel2_Num = kernel2_Num;
	this->classification_Num = classification_Num;
	std::cout << useData_Num << std::endl;
	op->MnistImageMatDataRead("Resources/train-images.idx3-ubyte", imageMats, 0,useData_Num);
	op->MnistImageLabelDataRead("Resources/train-labels.idx1-ubyte", imageLabels, 0,useData_Num);

	SplitData(imageMats, imageLabels, 16, 4, 5);

//	for (int i = 0; i < imageMats.size(); i++) {
//		trainingMats.push_back(cv::Mat());
//		trainingConv1x1ZeroPaddingMats.push_back(cv::Mat_<double>());
//
//		//uint8_t형 행렬 요소를 double형 행렬 요소로 타입 캐스팅
//		imageMats[i].convertTo(trainingMats[i], CV_64FC1);
//		//평균을 계산해 간단한 특성 스케일 정규화
//		trainingMats[i] /= 255;
//	}
//
//#pragma region 정답 데이터 행렬과 가설 행렬 초기화
//	//정답 데이터를 벡터로 변환한다.
//	trainingYMat = cv::Mat::zeros(cv::Size(classification_Num, trainingMats.size()), CV_64FC1);
//	for (int y = 0; y < labelVec.size(); y++) {
//		//열과 맞는다면 true(1), 아니라면 false(0)를 저장
//		trainingYMat.at<double>(y, labelVec[y]) = 1;
//	}
//
//	trainingNeuralYHatMat.create(cv::Size(classification_Num, trainingMats.size()), CV_64FC1);
//#pragma endregion
	poolStride = cv::Size(2, 2);
	poolSize = cv::Size(2, 2);

#pragma region 모든 가중치 행렬을 균등 분포로 랜덤 초기화, 커널 역방향 계산 필터 초기화
	cv::RNG gen(cv::getTickCount());
	//커널 1은 채널 한개(입력층 채널이 흑백 단일)
	
	kernels1.push_back(std::vector<cv::Mat>());
	conv1ResultBiases.push_back(std::vector<double>());

	for (int k1i = 0; k1i < kernel1_Num; k1i++) {
		kernels1[0].push_back(cv::Mat(cv::Size(3, 3), CV_64FC1));
		conv1ResultBiases[0].push_back(0);

		gen.fill(kernels1[0][k1i], cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(2));

		kernels2.push_back(std::vector<cv::Mat>());
		conv2ResultBiases.push_back(std::vector<double>());

		//커널 2는 채널이 커널 1의 개수
		for (int k2i = 0; k2i < kernel2_Num; k2i++) {
			kernels2[k1i].push_back(cv::Mat(cv::Size(3, 3), CV_64FC1));
			conv2ResultBiases[k1i].push_back(0);

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
	neuralW1Mat.create(cv::Size(10, wHeight * wWidth * kernel2_Num), CV_64FC1);
	gen.fill(neuralW1Mat, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(1));
	neuralW2.create(cv::Size(classification_Num, neuralW1Mat.cols), CV_64FC1);
	gen.fill(neuralW2, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(1));

	
#pragma endregion

#pragma region 합성곱 결과, 합성곱 결과 제로 패딩, 풀링 결과, 풀링 결과 제로 패딩, 풀링 역방향 계산 필터 행렬 초기화

	for (int i = 0; i < trainingMats.size(); i++) {
		trainingConv1ResultMats.push_back(std::vector<cv::Mat>());
		trainingConv2ResultMats.push_back(std::vector<cv::Mat>());
		trainingConv1ResultZeroPadMats.push_back(std::vector<cv::Mat>());
		trainingConv2ResultZeroPadMats.push_back(std::vector<cv::Mat>());
		trainingPool1Result.push_back(std::vector<cv::Mat>());
		trainingPool2Result.push_back(std::vector<cv::Mat>());
		trainingPool1ResultZeroPadding.push_back(std::vector<cv::Mat>());
		pool1BackpropFilters.push_back(std::vector<cv::Mat>());
		pool2BackpropFilters.push_back(std::vector<cv::Mat>());
		neuralBiases1.push_back(0);
		neuralBiases2.push_back(0);
		for (int j = 0; j < kernel1_Num; j++) {
			trainingConv1ResultMats[i].push_back(cv::Mat_<double>());
			trainingConv1ResultZeroPadMats[i].push_back(cv::Mat_<double>());
			trainingPool1Result[i].push_back(cv::Mat_<double>());
			trainingPool1ResultZeroPadding[i].push_back(cv::Mat_<double>());
		
			pool1BackpropFilters[i].push_back(cv::Mat_<double>());
		}
		for (int j = 0; j < kernel2_Num; j++) {
			trainingConv2ResultMats[i].push_back(cv::Mat_<double>());
			trainingConv2ResultZeroPadMats[i].push_back(cv::Mat_<double>());
			trainingPool2Result[i].push_back(cv::Mat_<double>());
			
			pool2BackpropFilters[i].push_back(cv::Mat_<double>());
		}
	}
#pragma endregion

#pragma region 합성곱 역방향 사용 변수 초기화
	//합성곱 시 세임 패딩만 사용하므로 풀링 결과 크기만 계산
	trainingPool1ResultSize =
		cv::Size(
			(trainingMats[0].size().width - poolSize.width) / poolStride.width + 1,
			(trainingMats[0].size().height - poolSize.height) / poolStride.height + 1
		);
	trainingPool2ResultSize =
		cv::Size(
			(trainingPool1ResultSize.width - poolSize.width) / poolStride.width + 1,
			(trainingPool1ResultSize.height - poolSize.height) / poolStride.height + 1
		);
	//1번째 합성곱의 역방향 계산 필터 초기화
	int r1Size = trainingMats[0].rows * trainingMats[0].cols;
	for (int r1i = 0; r1i < r1Size; r1i++) {
		conv1BackpropFilters.push_back(std::pair<int, int>());
	}
	//2번째 합성곱 커널의 역방향 계산 필터 초기화
	int r2Size = trainingPool1ResultSize.width * trainingPool1ResultSize.height;
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



#pragma region 모델 예측 사용 변수 초기화
	for (int j = 0; j < kernel1_Num; j++) {
		predictConv1Mats.push_back(cv::Mat_<double>());
		predictConv1ZeroPaddingMats.push_back(cv::Mat_<double>());
		predictPool1Result.push_back(cv::Mat_<double>());
		predictPool1ResultZeroPadding.push_back(cv::Mat_<double>());
	}
	for (int j = 0; j < kernel2_Num; j++) {
		predictConv2Mats.push_back(cv::Mat_<double>());
		predictConv2ZeroPaddingMats.push_back(cv::Mat_<double>());
		predictPool2Result.push_back(cv::Mat_<double>());
	}
#pragma endregion
}

void CNNMachine::ForwardPropagationStochastic(int trainingIndex)
{
}

void CNNMachine::BackPropagationStochastic(int trainingIndex)
{
	//yLoss = -(trainingYMat - trainingNeuralYHatMat); //손실함수를 SoftMax 함수 결과에 대해 미분한 값
	yLoss = trainingNeuralYHatMat - trainingYMat; //손실함수를 SoftMax 함수 결과에 대해 미분한 값
	w2T = neuralW2.t();
	yLossW2 = yLoss * w2T; //손실 함수를 완전연결층2 입력(ReLu(aMat))에 대해 미분한 값

	//std::cout << "yLossW2\n" << yLossW2 <<std::endl;
	//Relu(trainingNeuralX2)과 벡터곱
	//(정방향 계산에서 이미 ReLU를 적용했으므로 생략)
	//Math::Relu(trainingNeuralX2, trainingNeuralX2);
	yLossW2Relu3 = yLossW2.mul(trainingNeuralX2); //손실함수를 완전연결층1 결과에 대해 미분한 값

	//std::cout << "Relu(trainingNeuralX2)\n" << trainingNeuralX2 << std::endl;
	//std::cout << "Relu(trainingNeuralX2)과 벡터곱\n" << yLossW2Relu3 << std::endl;
	yLossW2Relu3W1 = yLossW2Relu3 * neuralW1Mat.t();

	//yLossW2Relu3W1를 합성곱층 결과 크기로 차원 변환, 풀링2 필터로 Up-Sampleling, Relu(Conv2)행렬과 벡터곱
	int shuffledIndex = trainingIndex;
	for (int k2n = 0; k2n < kernel2_Num; k2n++) {
		//Pooling 함수 역방향 계산으로 풀링 필터 할당
		Math::GetMaxPoolingFilter(trainingConv2ResultZeroPadMats[shuffledIndex][k2n], pool2BackpropFilters[shuffledIndex][k2n], trainingPool2Result[shuffledIndex][k2n], poolSize, poolStride);
		//풀링 필터로 업샘플링
		cv::Mat sample = yLossW2Relu3W1.row(shuffledIndex).reshape(1, trainingPool2Result[0].size()).row(k2n).reshape(1, trainingPool2Result[0][0].rows);
		Math::MaxPoolingBackprop(sample, yLossW2Relu3W1UpRelu2[shuffledIndex][k2n], pool2BackpropFilters[shuffledIndex][k2n], poolSize, poolStride);

		//Relu 함수 역방향 계산
		//Up-Sampleling 결과 행렬과 Relu(Conv2)행렬을 벡터곱
		yLossW2Relu3W1UpRelu2[shuffledIndex][k2n] = yLossW2Relu3W1UpRelu2[shuffledIndex][k2n].mul(trainingConv2ResultMats[shuffledIndex][k2n]);

	}
	//std::cout << "yLossW2Relu3W1UpRelu2[0][0]\n" << yLossW2Relu3W1UpRelu2[0][0] << std::endl;

	//커널2 역방향 계산을 위한 합성곱2 필터 계산
	Math::GetConvBackpropFilters(trainingPool1Result[0][0], &conv2BackpropFilters, kernels2[0][0], kernel2Stride);
	//커널1 역방향 계산을 위한 합성곱1 필터 계산
	Math::GetConvBackpropFilters(trainingMats[0], &conv1BackpropFilters, kernels1[0][0], kernel1Stride);

#pragma region 합성곱층1 가중치 행렬(커널1), 편향 역방향 계산
	//yLossW2Relu3W1UpRelu2행렬과 합성곱2 함수의 커널2에 대한 미분 행렬을 벡터곱하고, 풀링1 필터로 Up-Sampleling 후 Relu(Conv1)행렬과 벡터곱
	//커널 1 개수만큼 반복
	for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
		cv::Mat yLossW2Relu3W1UpRelu2P1 = cv::Mat(yLossW2Relu3W1UpRelu2[shuffledIndex][0].size(), CV_64FC1);
		yLossW2Relu3W1UpRelu2P1.setTo(0);

		//커널 2 개수만큼 반복
		for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
			cv::Mat conv2XBackprop;
			//yLossW2Relu3W1UpRelu2행렬과 합성곱2 함수의 커널2에 대한 미분 행렬을 벡터곱
			Math::ConvXBackprop(yLossW2Relu3W1UpRelu2[shuffledIndex][k2n], kernels2[k1n][k2n], conv2XBackprop, conv2BackpropFilters, kernel1Stride);
			yLossW2Relu3W1UpRelu2P1 += yLossW2Relu3W1UpRelu2[shuffledIndex][k2n].mul(conv2XBackprop);
		}
		//평균 계산으로 특성 스케일 정규화
		yLossW2Relu3W1UpRelu2P1 /= kernels2[0].size();

		//Pooling 함수 역방향 계산으로 풀링 필터 정의
		Math::GetMaxPoolingFilter(trainingConv1ResultZeroPadMats[shuffledIndex][k1n], pool1BackpropFilters[shuffledIndex][k1n], trainingPool1Result[shuffledIndex][k1n], poolSize, poolStride);
		//풀링 필터로 업샘플링
		Math::MaxPoolingBackprop(yLossW2Relu3W1UpRelu2P1, yLossW2Relu3W1UpRelu2P1UpRelu[shuffledIndex][k1n], pool1BackpropFilters[shuffledIndex][k1n], poolSize, poolStride);

		//Relu 함수 역방향 계산
		//(정방향 계산에서 이미 ReLU를 적용했으므로 생략)
		//Math::Relu(trainingConv1ResultMats[shuffledIndex][k1n], trainingConv1ResultMats[shuffledIndex][k1n]);
		yLossW2Relu3W1UpRelu2P1UpRelu[shuffledIndex][k1n] = yLossW2Relu3W1UpRelu2P1UpRelu[shuffledIndex][k1n].mul(trainingConv1ResultMats[shuffledIndex][k1n]);
	}

	for (int k1c = 0; k1c < kernels1.size(); k1c++) {
		for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
			cv::Mat conv1KBackprops = cv::Mat::zeros(kernels1[0][0].size(), CV_64FC1);
			double conv1BiasBackprop = 0;

			cv::Mat conv1KBackpropTemp;
			Math::ConvKBackprop(yLossW2Relu3W1UpRelu2P1UpRelu[shuffledIndex][k1n], trainingConv1x1ZeroPaddingMats[shuffledIndex], kernels1[0][0].size(), conv1KBackpropTemp, conv1BackpropFilters, kernel1Stride);
			conv1KBackprops += conv1KBackpropTemp;

			for (int bY = 0; bY < trainingConv1ResultMats[0][0].rows; bY++) {
				for (int bX = 0; bX < trainingConv1ResultMats[0][0].cols; bX++) {
					conv1BiasBackprop += yLossW2Relu3W1UpRelu2P1UpRelu[shuffledIndex][k1n].at<double>(bY, bX);
				}
			}
			//행렬 요소를 전부 더한 후, 요소 갯수만큼 나눈다. 특성 스케일 정규화 후 편향을 업데이트
			conv1ResultBiases[k1c][k1n] -= learningRate * conv1BiasBackprop / (trainingConv1ResultMats[0][0].rows * trainingConv1ResultMats[0][0].cols * trainingMats.size());

			kernels1[k1c][k1n] -= learningRate * conv1KBackprops / trainingMats.size();
		}
	}
	//정규화
	/*for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
		for (int k1c = 0; k1c < kernels1.size(); k1c++) {
			kernels1[k1c][k1n] /= trainingMats.size();
		}
	}*/

#pragma endregion

#pragma region 합성곱층2 가중치 행렬(커널2), 편향 역방향 계산
	for (int k2c = 0; k2c < kernels2.size(); k2c++) {
		for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
			cv::Mat conv2KBackprops = cv::Mat::zeros(kernels2[0][0].size(), CV_64FC1);
			double conv2BiasBackprop = 0;

			cv::Mat conv2KBackpropTemp;
			Math::ConvKBackprop(yLossW2Relu3W1UpRelu2[shuffledIndex][k2n], trainingPool1ResultZeroPadding[shuffledIndex][k2c], kernels2[0][0].size(), conv2KBackpropTemp, conv2BackpropFilters, kernel2Stride);
			conv2KBackprops += conv2KBackpropTemp;

			for (int bY = 0; bY < trainingConv2ResultMats[0][0].rows; bY++) {
				for (int bX = 0; bX < trainingConv2ResultMats[0][0].cols; bX++) {
					conv2BiasBackprop += yLossW2Relu3W1UpRelu2[shuffledIndex][k2n].at<double>(bY, bX);
				}
			}
			//행렬 요소를 전부 더한 후, 요소 갯수만큼 나눈다. 특성 스케일 정규화 후 편향을 업데이트
			conv2ResultBiases[k2c][k2n] -= learningRate * conv2BiasBackprop / (trainingConv2ResultMats[0][0].rows * trainingConv2ResultMats[0][0].cols * trainingMats.size());
			kernels2[k2c][k2n] -= learningRate * conv2KBackprops / trainingMats.size();
		}
	}
	//정규화
	/*for (int k2n = 0; k2n < kernels1[0].size(); k2n++) {
		for (int k2c = 0; k2c < kernels1.size(); k2c++) {
			kernels2[k2c][k2n] /= trainingMats.size();
		}
	}*/

#pragma endregion

#pragma region 완전연결신경망층 가중치 행렬, 편향 역방향 계산
	neuralW1Mat -= learningRate * trainingNeuralX1Mat.t() * (yLossW2Relu3) / yLossW2Relu3.rows;//평균을 계산해 간단한 특성 스케일 정규화
	for (int y = 0; y < yLossW2Relu3.rows; y++) {
		double bias1Backprop = 0;
		for (int x = 0; x < yLossW2Relu3.cols; x++) {
			bias1Backprop += yLossW2Relu3.at<double>(y, x);
		}
		neuralBiases1[y] -= learningRate * bias1Backprop / yLossW2Relu3.cols;
	}

	neuralW2 -= learningRate * (trainingNeuralX2.t() * (yLoss)) / yLoss.rows;//평균을 계산해 간단한 특성 스케일 정규화
	for (int y = 0; y < yLoss.rows; y++) {
		double bias2Backprop = 0;
		for (int x = 0; x < yLoss.cols; x++) {
			bias2Backprop += yLoss.at<double>(y, x);
		}
		neuralBiases2[y] -= learningRate * bias2Backprop / yLoss.cols;
	}
#pragma endregion
}

void CNNMachine::ForwardPropagationMiniBatch(int miniBatchIndex)
{
}

void CNNMachine::BackPropagationMiniBatch(int miniBatchIndex)
{
}

void CNNMachine::ForwardPropagationBatch()
{
	//합성곱층 결과 행렬을 완전연결신경망 입력으로 변환할 때 사용
	std::vector<std::vector<double>> tempArr;
	cv::Mat tempMat;

	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		tempArr.push_back(std::vector<double>());

		Math::CreateZeroPadding(trainingMats[x1i], trainingConv1x1ZeroPaddingMats[x1i], trainingMats[0].size(), kernels1[0][0].size(), kernel1Stride);
		//합성곱층 1
		for (int k1n = 0; k1n < kernel1_Num; k1n++) {
			Math::Convolution(trainingConv1x1ZeroPaddingMats[x1i], trainingConv1ResultMats[x1i][k1n], trainingMats[0].size(), kernels1[0][k1n], kernel1Stride);
			trainingConv1ResultMats[x1i][k1n] += conv1ResultBiases[0][k1n];
			Math::Relu(trainingConv1ResultMats[x1i][k1n], trainingConv1ResultMats[x1i][k1n]);
			Math::CreateZeroPadding(trainingConv1ResultMats[x1i][k1n], trainingConv1ResultZeroPadMats[x1i][k1n], trainingPool1ResultSize, poolSize, poolStride);
			Math::MaxPooling(trainingConv1ResultZeroPadMats[x1i][k1n], trainingPool1Result[x1i][k1n], poolSize, poolStride);

			Math::CreateZeroPadding(trainingPool1Result[x1i][k1n], trainingPool1ResultZeroPadding[x1i][k1n], trainingPool1ResultSize, kernels2[0][0].size(), kernel2Stride);
		}
		//합성곱층 2
		/*합성곱층 1의 (행:데이터 수, 열:채널 수)의 이미지을 가진 trainingPool1Result행렬과
		합성곱층 2의 kernel2행렬을 행렬곱하듯 연결*/
		for (int k2n = 0; k2n < kernel2_Num; k2n++) {
			//double conv2ResultBiasesSum = 0;
			for (int k1n = 0; k1n < kernel1_Num; k1n++) {
				Math::Convolution(trainingPool1ResultZeroPadding[x1i][k1n], trainingConv2ResultMats[x1i][k2n], trainingPool1ResultSize, kernels2[k1n][k2n], kernel2Stride);
				//conv2ResultBiasesSum += conv2ResultBiases[k1n][k2n];
				trainingConv2ResultMats[x1i][k2n] += conv2ResultBiases[k1n][k2n];
				
			}
			//trainingConv2ResultMats[x1i][k2n] += conv2ResultBiasesSum / kernel1_Num;
			Math::Relu(trainingConv2ResultMats[x1i][k2n], trainingConv2ResultMats[x1i][k2n]);
			Math::CreateZeroPadding(trainingConv2ResultMats[x1i][k2n], trainingConv2ResultZeroPadMats[x1i][k2n], trainingPool2ResultSize, poolSize, poolStride);
			Math::MaxPooling(trainingConv2ResultZeroPadMats[x1i][k2n], trainingPool2Result[x1i][k2n], poolSize, poolStride);
		}
		//완전연결신경망 입력
		//4차원 trainingPool2Result를 2차원 행렬 trainingNeuralX1Mat으로 변환
		//vec<vec<Mat>> to vec<vec<double>> 변환 : https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
		//vec<vec<double>> to Mat 변환 : https://stackoverflow.com/questions/18519647/opencv-convert-vector-of-vector-to-mat
		for (int k2n = 0; k2n < kernel2_Num; k2n++) {
			tempMat = trainingPool2Result[x1i][k2n];
			for (int i = 0; i < tempMat.rows; ++i) {
				tempArr[x1i].insert(tempArr[x1i].end(), tempMat.ptr<double>(i), tempMat.ptr<double>(i) + tempMat.cols * tempMat.channels());
			}
		}
	}

	trainingNeuralX1Mat.create(cv::Size(0, tempArr[0].size()), CV_64FC1);

	//훈련 데이터 수만큼 반복
	for (int i = 0; i < tempArr.size(); ++i)
	{
		//Make a temporary cv::Mat row and add to NewSamples _without_ data copy
		cv::Mat Sample(1, tempArr[0].size(), CV_64FC1, tempArr[i].data());
		trainingNeuralX1Mat.push_back(Sample);
	}
	Math::NeuralNetwork(trainingNeuralX1Mat, trainingNeuralX2, neuralW1Mat);
	//trainingNeuralX2 /= trainingNeuralX1Mat.cols;//특성 스케일 정규화
	for (int a1i = 0; a1i < trainingNeuralX2.rows; a1i++) {
		trainingNeuralX2.row(a1i) += neuralBiases1[a1i];
	}

	Math::Relu(trainingNeuralX2, trainingNeuralX2);

	Math::NeuralNetwork(trainingNeuralX2, trainingNeuralYHatMat, neuralW2);
	//trainingNeuralYHatMat /= trainingNeuralX2.cols;//특성 스케일 정규화

	for (int yHati = 0; yHati < trainingNeuralYHatMat.rows; yHati++) {
		trainingNeuralYHatMat.row(yHati) += neuralBiases2[yHati];
	}
	Math::SoftMax(trainingNeuralYHatMat, trainingNeuralYHatMat);
}

void CNNMachine::BackPropagationBatch()
{
	//yLoss = -(trainingYMat - trainingNeuralYHatMat); //손실함수를 SoftMax 함수 결과에 대해 미분한 값
	yLoss = trainingNeuralYHatMat - trainingYMat; //손실함수를 SoftMax 함수 결과에 대해 미분한 값
	w2T = neuralW2.t();
	yLossW2 = yLoss*w2T; //손실 함수를 완전연결층2 입력(ReLu(aMat))에 대해 미분한 값
	
	//std::cout << "yLossW2\n" << yLossW2 <<std::endl;
	//Relu(trainingNeuralX2)과 벡터곱
	//(정방향 계산에서 이미 ReLU를 적용했으므로 생략)
	//Math::Relu(trainingNeuralX2, trainingNeuralX2);
	yLossW2Relu3 = yLossW2.mul(trainingNeuralX2); //손실함수를 완전연결층1 결과에 대해 미분한 값
	
	//std::cout << "Relu(trainingNeuralX2)\n" << trainingNeuralX2 << std::endl;
	//std::cout << "Relu(trainingNeuralX2)과 벡터곱\n" << yLossW2Relu3 << std::endl;
	yLossW2Relu3W1 = yLossW2Relu3 * neuralW1Mat.t();

	//yLossW2Relu3W1를 합성곱층 결과 크기로 차원 변환, 풀링2 필터로 Up-Sampleling, Relu(Conv2)행렬과 벡터곱
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		for (int k2n = 0; k2n < kernel2_Num; k2n++) {
			//Pooling 함수 역방향 계산으로 풀링 필터 할당
			Math::GetMaxPoolingFilter(trainingConv2ResultZeroPadMats[x1i][k2n], pool2BackpropFilters[x1i][k2n], trainingPool2Result[x1i][k2n], poolSize, poolStride);
			//풀링 필터로 업샘플링
			cv::Mat sample = yLossW2Relu3W1.row(x1i).reshape(1, trainingPool2Result[0].size()).row(k2n).reshape(1, trainingPool2Result[0][0].rows);
			Math::MaxPoolingBackprop(sample, yLossW2Relu3W1UpRelu2[x1i][k2n], pool2BackpropFilters[x1i][k2n], poolSize, poolStride);

			//Relu 함수 역방향 계산
			//Up-Sampleling 결과 행렬과 Relu(Conv2)행렬을 벡터곱
			yLossW2Relu3W1UpRelu2[x1i][k2n] = yLossW2Relu3W1UpRelu2[x1i][k2n].mul(trainingConv2ResultMats[x1i][k2n]);
		}
	}
	//std::cout << "yLossW2Relu3W1UpRelu2[0][0]\n" << yLossW2Relu3W1UpRelu2[0][0] << std::endl;
	
	//커널2 역방향 계산을 위한 합성곱2 필터 계산
	Math::GetConvBackpropFilters(trainingPool1Result[0][0], &conv2BackpropFilters, kernels2[0][0], kernel2Stride);
	//커널1 역방향 계산을 위한 합성곱1 필터 계산
	Math::GetConvBackpropFilters(trainingMats[0], &conv1BackpropFilters, kernels1[0][0], kernel1Stride);

#pragma region 합성곱층1 가중치 행렬(커널1), 편향 역방향 계산
	//yLossW2Relu3W1UpRelu2행렬과 합성곱2 함수의 커널2에 대한 미분 행렬을 벡터곱하고, 풀링1 필터로 Up-Sampleling 후 Relu(Conv1)행렬과 벡터곱
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		//커널 1 개수만큼 반복
		for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
			cv::Mat yLossW2Relu3W1UpRelu2P1 = cv::Mat(yLossW2Relu3W1UpRelu2[x1i][0].size(), CV_64FC1);
			yLossW2Relu3W1UpRelu2P1.setTo(0);

			//커널 2 개수만큼 반복
			for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
				cv::Mat conv2XBackprop;
				//yLossW2Relu3W1UpRelu2행렬과 합성곱2 함수의 커널2에 대한 미분 행렬을 벡터곱
				Math::ConvXBackprop(yLossW2Relu3W1UpRelu2[x1i][k2n], kernels2[k1n][k2n], conv2XBackprop, conv2BackpropFilters, kernel1Stride);
				yLossW2Relu3W1UpRelu2P1 += yLossW2Relu3W1UpRelu2[x1i][k2n].mul(conv2XBackprop);
			}
			//평균 계산으로 특성 스케일 정규화
			yLossW2Relu3W1UpRelu2P1 /= kernels2[0].size();

			//Pooling 함수 역방향 계산으로 풀링 필터 정의
			Math::GetMaxPoolingFilter(trainingConv1ResultZeroPadMats[x1i][k1n], pool1BackpropFilters[x1i][k1n], trainingPool1Result[x1i][k1n], poolSize, poolStride);
			//풀링 필터로 업샘플링
			Math::MaxPoolingBackprop(yLossW2Relu3W1UpRelu2P1, yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n], pool1BackpropFilters[x1i][k1n], poolSize, poolStride);

			//Relu 함수 역방향 계산
			//(정방향 계산에서 이미 ReLU를 적용했으므로 생략)
			//Math::Relu(trainingConv1ResultMats[x1i][k1n], trainingConv1ResultMats[x1i][k1n]);
			yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n] = yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n].mul(trainingConv1ResultMats[x1i][k1n]);
		}
	}
	for (int k1c = 0; k1c < kernels1.size(); k1c++) {
		for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
			cv::Mat conv1KBackprops = cv::Mat::zeros(kernels1[0][0].size(), CV_64FC1);
			double conv1BiasBackprop = 0;

			for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
				cv::Mat conv1KBackpropTemp;
				Math::ConvKBackprop(yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n], trainingConv1x1ZeroPaddingMats[x1i], kernels1[0][0].size(), conv1KBackpropTemp, conv1BackpropFilters, kernel1Stride);
				conv1KBackprops += conv1KBackpropTemp;

				for (int bY = 0; bY < trainingConv1ResultMats[0][0].rows; bY++) {
					for (int bX = 0; bX < trainingConv1ResultMats[0][0].cols; bX++) {
						conv1BiasBackprop += yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n].at<double>(bY, bX);
					}
				}
				
			}
			//행렬 요소를 전부 더한 후, 요소 갯수만큼 나눈다. 특성 스케일 정규화 후 편향을 업데이트
			conv1ResultBiases[k1c][k1n] -= learningRate * conv1BiasBackprop / (trainingConv1ResultMats[0][0].rows * trainingConv1ResultMats[0][0].cols * trainingMats.size());

			kernels1[k1c][k1n] -= learningRate * conv1KBackprops / trainingMats.size();
		}
	}
	//정규화
	/*for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
		for (int k1c = 0; k1c < kernels1.size(); k1c++) {
			kernels1[k1c][k1n] /= trainingMats.size();
		}
	}*/

#pragma endregion

#pragma region 합성곱층2 가중치 행렬(커널2), 편향 역방향 계산
	for (int k2c = 0; k2c < kernels2.size(); k2c++) {
		for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
			cv::Mat conv2KBackprops = cv::Mat::zeros(kernels2[0][0].size(), CV_64FC1);
			double conv2BiasBackprop = 0;

			for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
				cv::Mat conv2KBackpropTemp;
				Math::ConvKBackprop(yLossW2Relu3W1UpRelu2[x1i][k2n], trainingPool1ResultZeroPadding[x1i][k2c], kernels2[0][0].size(), conv2KBackpropTemp, conv2BackpropFilters, kernel2Stride);
				conv2KBackprops += conv2KBackpropTemp;

				for (int bY = 0; bY < trainingConv2ResultMats[0][0].rows; bY++) {
					for (int bX = 0; bX < trainingConv2ResultMats[0][0].cols; bX++) {
						conv2BiasBackprop += yLossW2Relu3W1UpRelu2[x1i][k2n].at<double>(bY, bX);
					}
				}
			}
			//행렬 요소를 전부 더한 후, 요소 갯수만큼 나눈다. 특성 스케일 정규화 후 편향을 업데이트
			conv2ResultBiases[k2c][k2n] -= learningRate * conv2BiasBackprop / (trainingConv2ResultMats[0][0].rows * trainingConv2ResultMats[0][0].cols * trainingMats.size());
			kernels2[k2c][k2n] -= learningRate * conv2KBackprops / trainingMats.size();
		}
	}
	//정규화
	/*for (int k2n = 0; k2n < kernels1[0].size(); k2n++) {
		for (int k2c = 0; k2c < kernels1.size(); k2c++) {
			kernels2[k2c][k2n] /= trainingMats.size();
		}
	}*/

#pragma endregion

#pragma region 완전연결신경망층 가중치 행렬, 편향 역방향 계산
	neuralW1Mat -= learningRate * trainingNeuralX1Mat.t() * (yLossW2Relu3) / yLossW2Relu3.rows;//평균을 계산해 간단한 특성 스케일 정규화
	for (int y = 0; y < yLossW2Relu3.rows; y++) {
		double bias1Backprop = 0;
		for (int x = 0; x < yLossW2Relu3.cols; x++) {
			bias1Backprop += yLossW2Relu3.at<double>(y, x);
		}
		neuralBiases1[y] -= learningRate * bias1Backprop / yLossW2Relu3.cols;
	}
	
	neuralW2 -= learningRate * (trainingNeuralX2.t() * (yLoss)) / yLoss.rows;//평균을 계산해 간단한 특성 스케일 정규화
	for (int y = 0; y < yLoss.rows; y++) {
		double bias2Backprop = 0;
		for (int x = 0; x < yLoss.cols; x++) {
			bias2Backprop += yLoss.at<double>(y, x);
		}
		neuralBiases2[y] -= learningRate * bias2Backprop / yLoss.cols;
	}
#pragma endregion
}

void CNNMachine::ModelPredict(cv::InputArray _Input, cv::OutputArray _NeuralYHatMatOutput)
{
	//합성곱층 결과 행렬을 완전연결신경망 입력으로 변환할 때 사용
	std::vector<double> tempArr;
	cv::Mat tempMat;

	
	cv::Mat input = _Input.getMat();
	//std::cout << input << std::endl;
	
	//int i = rand() % useData_Num;
	//cv::Mat input = trainingMats[i];
	//cv::imshow("Debug", input);
	cv::Mat inputZeroPad;

	Math::CreateZeroPadding(input, inputZeroPad, input.size(), kernels1[0][0].size(), kernel1Stride);

	//합성곱층 1
	for (int k1n = 0; k1n < kernel1_Num; k1n++) {
		Math::Convolution(inputZeroPad, predictConv1Mats[k1n], trainingMats[0].size(), kernels1[0][k1n], kernel1Stride);
		predictConv1Mats[k1n] += conv1ResultBiases[0][k1n];

		Math::Relu(predictConv1Mats[k1n], predictConv1Mats[k1n]);
		Math::CreateZeroPadding(predictConv1Mats[k1n], predictConv1ZeroPaddingMats[k1n], trainingPool1ResultSize, poolSize, poolStride);
		Math::MaxPooling(predictConv1ZeroPaddingMats[k1n], predictPool1Result[k1n], poolSize, poolStride);

		Math::CreateZeroPadding(predictPool1Result[k1n], predictPool1ResultZeroPadding[k1n], trainingPool1ResultSize, kernels2[0][0].size(), kernel2Stride);
	}

	//합성곱층 2
	/*합성곱층 1의 (행:데이터 수, 열:채널 수)의 이미지을 가진 trainingPool1Result행렬과
	합성곱층 2의 kernel2행렬을 행렬곱하듯 연결*/
	for (int k2n = 0; k2n < kernel2_Num; k2n++) {
		for (int k1n = 0; k1n < kernel1_Num; k1n++) {
			Math::Convolution(predictPool1ResultZeroPadding[k1n], predictConv2Mats[k2n], trainingPool1ResultSize, kernels2[k1n][k2n], kernel2Stride);
			predictConv2Mats[k2n] += conv2ResultBiases[k1n][k2n];
		}
		//std::cout << predictConv2Mats[k2n] << std::endl;
		Math::Relu(predictConv2Mats[k2n], predictConv2Mats[k2n]);
		Math::CreateZeroPadding(predictConv2Mats[k2n], predictConv2ZeroPaddingMats[k2n], trainingPool2ResultSize, poolSize, poolStride);
		Math::MaxPooling(predictConv2ZeroPaddingMats[k2n], predictPool2Result[k2n], poolSize, poolStride);
	}

	//완전연결신경망 입력
	//4차원 trainingPool2Result를 2차원 행렬 trainingNeuralX1Mat으로 변환
	//vec<vec<Mat>> to vec<vec<double>> 변환 : https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
	//vec<vec<double>> to Mat 변환 : https://stackoverflow.com/questions/18519647/opencv-convert-vector-of-vector-to-mat
	for (int k2n = 0; k2n < kernel2_Num; k2n++) {
		tempMat = predictPool2Result[k2n];
		for (int i = 0; i < tempMat.rows; ++i) {
			tempArr.insert(tempArr.end(), tempMat.ptr<double>(i), tempMat.ptr<double>(i) + tempMat.cols * tempMat.channels());
		}
	}

	//cv::Mat predictNeuralInputMat;
	//predictNeuralInputMat.create(cv::Size(0, tempArr.size()), CV_64FC1);

	//cv::Mat Sample(1, tempArr.size(), CV_64FC1, tempArr.data());
	//predictNeuralX1Mat.push_back(Sample);
	predictNeuralX1Mat = cv::Mat(1, tempArr.size(), CV_64FC1, tempArr.data());
	Math::NeuralNetwork(predictNeuralX1Mat, predictNeuralX2Mat, neuralW1Mat);
	double neuralBiases1Temp = 0;
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		neuralBiases1Temp += neuralBiases1[x1i];
	}
	//편향의 평균을 더함
	predictNeuralX2Mat += neuralBiases1Temp / trainingMats.size();

	Math::Relu(predictNeuralX2Mat, predictNeuralX2Mat);

	Math::NeuralNetwork(predictNeuralX2Mat, predictNeuralYHatMat, neuralW2);

	//std::cout << predictNeuralYHatMat << std::endl;
	double neuralBiases2Temp = 0;
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		neuralBiases2Temp += neuralBiases2[x1i];
	}
	//편향의 평균을 더함
	predictNeuralYHatMat += neuralBiases2Temp / trainingMats.size();

	Math::SoftMax(predictNeuralYHatMat, predictNeuralYHatMat);

	
	predictNeuralYHatMat.copyTo(_NeuralYHatMatOutput);
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
	fs << "GradientDescent"		<< gradientDescent;
	fs << "PoolSize"			<< poolSize;
	fs << "PoolStride"			<< poolStride;
	fs << "Kernel1Stride"		<< kernel1Stride;
	fs << "Kernel2Stride"		<< kernel2Stride;

	fs << "Kernels1"			<< kernels1;
	fs << "Kernels2"			<< kernels2;
	fs << "Conv1Bias"			<< conv1ResultBiases;
	fs << "Conv2Bias"			<< conv2ResultBiases;
	fs << "neuralW1Mat"				<< neuralW1Mat;
	fs << "neuralW2"				<< neuralW2;
	fs << "Bias1"				<< neuralBiases1;
	fs << "Bias2"				<< neuralBiases2;

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
	fs["GradientDescent"]	>> gradientDescent;
	fs["PoolSize"]			>> poolSize;
	fs["PoolStride"]		>> poolStride;
	fs["Kernel1Stride"]		>> kernel1Stride;
	fs["Kernel2Stride"]		>> kernel2Stride;

	ReleaseVectors();
	Init(op, useData_Num, kernel1_Num, kernel2_Num, classification_Num);
	fs["lossAverages"]		>> lossAverages;
	fs["Kernels1"]			>> kernels1;
	fs["Kernels2"]			>> kernels2;
	fs["Conv1Bias"]			>> conv1ResultBiases;
	fs["Conv2Bias"]			>> conv2ResultBiases;
	fs["neuralW1Mat"]				>> neuralW1Mat;
	fs["neuralW2"]				>> neuralW2;
	fs["Bias1"]				>> neuralBiases1;
	fs["Bias2"]				>> neuralBiases2;
	std::cout << kernels1.size() << std::endl;

	std::cout << "Model Load Completed!!!" << std::endl;
	fs.release();
	return true;
}

void CNNMachine::ReleaseVectors()
{
	lossAverages.clear();

	trainingMats.clear();
	trainingConv1x1ZeroPaddingMats.clear(); 
	kernels1.clear();
	kernels2.clear(); 
	trainingConv1ResultMats.clear();
	trainingConv2ResultMats.clear();
	conv1ResultBiases.clear();
	conv2ResultBiases.clear();
	trainingConv1ResultZeroPadMats.clear();
	trainingConv2ResultZeroPadMats.clear();
	trainingPool1Result.clear();
	trainingPool2Result.clear();
	trainingPool1ResultZeroPadding.clear();
	pool1BackpropFilters.clear();
	pool2BackpropFilters.clear(); 
	conv2BackpropFilters.clear();
	conv1BackpropFilters.clear();
	yLossW2Relu3W1UpRelu2.clear();
	yLossW2Relu3W1UpRelu2P1UpRelu.clear();
	predictConv1Mats.clear();
	predictConv1ZeroPaddingMats.clear();
	predictConv2Mats.clear();
	predictConv2ZeroPaddingMats.clear();
	predictPool1Result.clear();
	predictPool1ResultZeroPadding.clear();
	predictPool2Result.clear();

	neuralW1Mat.release();
	trainingNeuralX2.release();
}

bool CNNMachine::KeyEvent(int key)
{
	//입력 버퍼 초기화
	std::cin.clear();

	switch (key) {
	case 13: //enter키를 누르면 훈련 행렬 출력
	{
		std::cout << "정방향 계산에서 얻은 trainingYMat, trainingNeuralYHatMat, yLoss로 역방향 계산 끝. " << std::endl;
		//std::cout << "trainingYMat(정답 행렬) : " << std::endl;
		//std::cout << trainingYMat << std::endl;
		//std::cout << "trainingNeuralYHatMat(가설 행렬) : " << std::endl;
		//std::cout << trainingNeuralYHatMat << std::endl;
		std::cout << "yLoss (= -(trainingYMat - trainingNeuralYHatMat)) : " << std::endl;
		std::cout << yLoss << std::endl;
		std::cout << "kernels1[0][0] : " << std::endl;
		std::cout << kernels1[0][0] << std::endl;
		std::cout << "kernels2[0][0] : " << std::endl;
		std::cout << kernels2[0][0] << std::endl;
		std::cout << "conv1ResultBiases[0][0]" << std::endl;
		std::cout << conv1ResultBiases[0][0] << std::endl;
		std::cout << "conv1ResultBiases[0][1]" << std::endl;
		std::cout << conv1ResultBiases[0][1] << std::endl;

		std::cout << "conv2ResultBiases[0][0]" << std::endl;
		std::cout << conv2ResultBiases[0][0] << std::endl;
		std::cout << "conv2ResultBiases[0][1]" << std::endl;
		std::cout << conv2ResultBiases[0][1] << std::endl;
		std::cout << "neuralW1Mat[0][0]" << std::endl;
		std::cout << neuralW1Mat.at<double>(0, 0) << std::endl;
		std::cout << "neuralW2[0][0]" << std::endl;
		std::cout << neuralW2.at<double>(0, 0) << std::endl;
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

			PaintWindow(paintScreen, "Paint", paintScreen.size() * 200, 13);

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

void CNNMachine::PaintWindow(cv::InputOutputArray paintMat, cv::String windowName, cv::Size windowSize, int exitAsciiCode)
{
	cv::Mat inputScreen = paintMat.getMat();
	if (inputScreen.empty())
		inputScreen = cv::Mat::zeros(cv::Size(28, 28), CV_64FC1);
	inputScreen.setTo(0);

	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::imshow(windowName, inputScreen);
	cv::setMouseCallback(windowName, CallBackFunc, &inputScreen);
	cv::resizeWindow(windowName, windowSize);

	bool updateFlag = mouseLeftPress || mouseRightPress;
	std::cout << "종료하려면 Enter키를 누르십시오" << std::endl;

	do {
		predictTickMeter.reset();
		predictTickMeter.start();
		cv::imshow(windowName, inputScreen);
		if (updateFlag) {
			system("cls");
			std::cout << "종료하려면 Enter키를 누르십시오" << std::endl;
			ModelPredict(inputScreen, predictNeuralYHatMat);
			std::cout << "손글씨 숫자 예측 완료" << std::endl;
			for (int x = 0; x < predictNeuralYHatMat.cols; x++) {
				std::cout << x + 1 << "일 확률 : " << predictNeuralYHatMat.at<double>(0, x) * 100 << "%" << std::endl;
			}
		}
		updateFlag = mouseLeftPress || mouseRightPress;

		predictTickMeter.stop();
	
	} //while (cv::waitKeyEx(predictTickMeter.getTimeMilli() <= 0 ? 1 : predictTickMeter.getTimeMilli() + 10) != exitAsciiCode);
	while (cv::waitKeyEx(predictTickMeter.getTimeMilli() + 10) != exitAsciiCode);
}

//static 변수 초기화 (http://egloos.zum.com/kaludin/v/2461942)
cv::Point CNNMachine::mousePt(0, 0);
bool CNNMachine::mouseLeftPress = false;
bool CNNMachine::mouseRightPress = false;

void CNNMachine::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	mousePt.x = x;
	mousePt.y = y;

	//람다 익명함수 사용
	auto drawCircle = [](int x, int y, void* userdata) {

		cv::Mat& img = *((cv::Mat*)(userdata)); // 1st cast it back, then deref
		//opencv 도형 그리기 함수 인자 : https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html
		//공식 문서 : https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html
		//cv::circle(img, cv::Point(x, y), 2, cv::Scalar(0.8), cv::FILLED, cv::LineTypes::LINE_4);
		cv::circle(img, cv::Point(x, y), 1, cv::Scalar(1), cv::FILLED, cv::LineTypes::LINE_4);
	};

	auto eraserCircle = [](int x, int y, void* userdata) {

		cv::Mat& img = *((cv::Mat*)(userdata)); // 1st cast it back, then deref
		//opencv 도형 그리기 함수 인자 : https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html
		//공식 문서 : https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html
		cv::circle(img, cv::Point(x, y), 2, cv::Scalar(0), cv::FILLED, cv::LineTypes::LINE_4);
	};

	switch (event)
	{
	case cv::EVENT_MOUSEMOVE:
	{
		if (mouseLeftPress == true)
		{
			drawCircle(x, y, userdata);
		}
		if (mouseRightPress == true) {
			eraserCircle(x, y, userdata);
		}
		break;
	}
	case cv::EVENT_LBUTTONDOWN:
	{
		mouseLeftPress = true;
		drawCircle(x, y, userdata);
		break;
	}
	case cv::EVENT_LBUTTONUP:
	{
		mouseLeftPress = false;
		break;
	}
	case cv::EVENT_RBUTTONDOWN:
		mouseRightPress = true;
		eraserCircle(x, y, userdata);
		break;
	case cv::EVENT_RBUTTONUP:
		mouseRightPress = false;
		break;
	}
}
