#include "CNNMachine.h"

void CNNMachine::Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec)
{
	for (int i = 0; i < trainingVec.size(); i++) {
		trainingMats.push_back(cv::Mat());
		//uchar형 행렬 요소를 float형 행렬 요소로 타입 캐스팅
		trainingVec[i].convertTo(trainingMats[i], CV_32FC1);
	}

	poolStride = cv::Size(2, 2);
	poolSize = cv::Size(2, 2);
#pragma region 모든 가중치 행렬을 균등 분포로 랜덤 초기화, 커널 역방향 계산 필터 초기화
	cv::RNG gen(cv::getTickCount());

	//커널 1은 채널 한개(입력층 채널이 흑백 단일)
	kernels1.push_back(std::vector<cv::Mat>());
	for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
		kernels1[0].push_back(cv::Mat(cv::Size(3, 3), CV_32FC1));
		gen.fill(kernels1[0][k1i], cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(1));

		kernels2.push_back(std::vector<cv::Mat>());
		//커널 2는 채널이 커널 1의 개수
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			kernels2[k1i].push_back(cv::Mat(cv::Size(3, 3), CV_32FC1));
			gen.fill(kernels2[k1i][k2i], cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(1));
		}
		
		
	}
	kernel1Stride = cv::Size(1, 1);
	kernel2Stride = cv::Size(1, 1);
	
	//합성곱은 세임 패딩으로 진행하므로 풀링층 2개에서의 축소만 계산
	int wHeight = (trainingMats[0].rows - poolSize.height) / poolStride.height + 1;
		wHeight = (wHeight - poolSize.height) / poolStride.height + 1;
	int wWidth = (trainingMats[0].cols - poolSize.width) / poolStride.width + 1;
		wWidth = (wWidth - poolSize.width) / poolStride.width + 1;

	wMat.create(cv::Size(CLASSIFICATIONNUM, wHeight * wWidth * KERNEL2_NUM), CV_32FC1);
	gen.fill(wMat, cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(1));

	
#pragma endregion
#pragma region 합성곱 결과, 합성곱 결과 제로 패딩, 풀링 결과, 풀링 결과 제로 패딩, 풀링 역방향 계산 필터 행렬 초기화
	for (int i = 0; i < trainingMats.size(); i++) {
		conv1Mats.push_back(std::vector<cv::Mat>());
		conv2Mats.push_back(std::vector<cv::Mat>());
		conv1ZeroPaddingMats.push_back(std::vector<cv::Mat>());
		conv2ZeroPaddingMats.push_back(std::vector<cv::Mat>());
		poolresult1.push_back(std::vector<cv::Mat>());
		poolresult2.push_back(std::vector<cv::Mat>());
		poolresult1ZeroPadding.push_back(std::vector<cv::Mat>());

		pool1BackpropFilters.push_back(std::vector<cv::Mat>());
		pool2BackpropFilters.push_back(std::vector<cv::Mat>());
		for (int j = 0; j < KERNEL1_NUM; j++) {
			conv1Mats[i].push_back(cv::Mat_<float>());
			conv1ZeroPaddingMats[i].push_back(cv::Mat_<float>());
			poolresult1[i].push_back(cv::Mat_<float>());
			poolresult1ZeroPadding[i].push_back(cv::Mat_<float>());
		
			pool1BackpropFilters[i].push_back(cv::Mat_<float>());
		}
		for (int j = 0; j < KERNEL2_NUM; j++) {
			conv2Mats[i].push_back(cv::Mat_<float>());
			conv2ZeroPaddingMats[i].push_back(cv::Mat_<float>());
			poolresult2[i].push_back(cv::Mat_<float>());
			
			pool2BackpropFilters[i].push_back(cv::Mat_<float>());
		}
	}
#pragma endregion

#pragma region 합성곱 역방향 계산 필터 초기화
	cv::Size pool1ResultSize =
		cv::Size(
			(trainingMats[0].size().width - poolSize.width) / poolStride.width + 1,
			(trainingMats[0].size().height - poolSize.height) / poolStride.height + 1
		);
	/*cv::Size pool2ResultSize =
		cv::Size(
			(pool1ResultSize.width - poolSize.width) / poolStride.width + 1,
			(pool1ResultSize.height - poolSize.height) / poolStride.height + 1
		);*/

	//1번째 합성곱의 역방향 계산 필터 초기화
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		conv1BackpropFilters.push_back(std::vector<std::vector<std::pair<int, int>>>());
		//커널 1은 채널 한개(입력층 채널이 흑백 단일)
		for (int k1c = 0; k1c < kernels1.size(); k1c++) {
			conv1BackpropFilters[x1i].push_back(std::vector<std::pair<int, int>>());
			//합성곱 1의 결과 행렬 크기만큼 초기화
			int r1Size = trainingMats[0].rows * trainingMats[0].cols;
			for (int r1i = 0; r1i < r1Size; r1i++) {
				conv1BackpropFilters[x1i][k1c].push_back(std::pair<int, int>());
			}
		}
	}
	//2번째 합성곱 커널의 역방향 계산 필터 초기화
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		conv2BackpropFilters.push_back(std::vector<std::vector<std::pair<int, int>>>());
		//커널 2은 채널 개수가 커널1의 개수
		for (int k2c = 0; k2c < kernels1[0].size(); k2c++) {
			conv2BackpropFilters[x1i].push_back(std::vector<std::pair<int, int>>());
			//합성곱 2의 결과 행렬 크기만큼 초기화 
			int r2Size = pool1ResultSize.width * pool1ResultSize.height;
			for (int r2i = 0; r2i < r2Size; r2i++) {
				conv2BackpropFilters[x1i][k2c].push_back(std::pair<int, int>());
			}
		}
	}
#pragma endregion

	//정답 데이터를 벡터로 변환한다.
	yMat = cv::Mat::zeros(cv::Size(CLASSIFICATIONNUM, trainingMats.size()), CV_32FC1);
	for (int y = 0; y < labelVec.size(); y++) {
		//열과 맞는다면 true(1), 아니라면 false(0)를 저장
		yMat.at<float>(y, labelVec[y]) = 1;
	}

	yHatMat.create(cv::Size(CLASSIFICATIONNUM, trainingMats.size()), CV_32FC1);


	//lossAverage = 2305843009213693951;
}

void CNNMachine::ForwardPropagation()
{
	//행렬 사이즈
	cv::Size trainingMatrixSize = trainingMats[0].size();
	cv::Size k1MatrixSize = kernels1[0][0].size();
	cv::Size k2MatrixSize = kernels2[0][0].size();
	//합성곱 시 세임 패딩만 사용하므로 풀링 결과만 계산
	cv::Size pool1ResultSize =
		cv::Size(
			(trainingMats[0].size().width - poolSize.width) / poolStride.width + 1,
			(trainingMats[0].size().height - poolSize.height) / poolStride.height + 1
		);
	cv::Size pool2ResultSize = 
		cv::Size(
		(pool1ResultSize.width - poolSize.width) / poolStride.width + 1,
		(pool1ResultSize.height - poolSize.height) / poolStride.height + 1
	);
	std::vector<std::vector<float>> tempArr;
	cv::Mat tempMat;

	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		tempArr.push_back(std::vector<float>());
		x1ZeroPaddingMats.push_back(cv::Mat_<float>());

		Math::CreateZeroPadding(trainingMats[x1i], x1ZeroPaddingMats[x1i], trainingMatrixSize, k1MatrixSize, kernel1Stride);

		//합성곱층 1
		for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
			Math::Convolution(x1ZeroPaddingMats[x1i], conv1Mats[x1i][k1i], trainingMatrixSize, kernels1[0][k1i], kernel1Stride);
			Math::Relu(conv1Mats[x1i][k1i], conv1Mats[x1i][k1i]);
			Math::CreateZeroPadding(conv1Mats[x1i][k1i], conv1ZeroPaddingMats[x1i][k1i], pool1ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv1ZeroPaddingMats[x1i][k1i], poolresult1[x1i][k1i], poolSize, poolStride);

			Math::CreateZeroPadding(poolresult1[x1i][k1i], poolresult1ZeroPadding[x1i][k1i], poolresult1[0][0].size(), k2MatrixSize, kernel2Stride);
		}
		//합성곱층 2
		/*합성곱층 1의 (행:데이터 수, 열:채널 수)의 이미지을 가진 poolresult1행렬과
		합성곱층 2의 kernel2행렬을 행렬곱하듯 연결*/
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
				Math::Convolution(poolresult1ZeroPadding[x1i][k1i], conv2Mats[x1i][k2i], poolresult1[0][0].size(), kernels2[k1i][k2i], kernel2Stride);
			}
			Math::Relu(conv2Mats[x1i][k2i], conv2Mats[x1i][k2i]);
			Math::CreateZeroPadding(conv2Mats[x1i][k2i], conv2ZeroPaddingMats[x1i][k2i], pool2ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv2ZeroPaddingMats[x1i][k2i], poolresult2[x1i][k2i], poolSize, poolStride);
			//Math::GetMaxPoolingFilter(conv2Mats[x1i][k2i], pool2BackpropFilters[x1i][k2i], poolresult2[x1i][k2i], poolSize, poolStride);
		}
		//완전연결신경망 입력
		//4차원 poolresult2를 2차원 행렬 xMat으로 변환
		//vec<vec<Mat>> to vec<vec<float>> 변환 : https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
		//vec<vec<float>> to Mat 변환 : https://stackoverflow.com/questions/18519647/opencv-convert-vector-of-vector-to-mat
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			tempMat = poolresult2[x1i][k2i];
			/*imshow("Window", trainingMats[x1i]);
			cv::namedWindow("Windowaa", cv::WINDOW_NORMAL);
			imshow("Windowaa", tempMat);
			if (cv::waitKey(0) != -1)
				continue;*/
			for (int i = 0; i < tempMat.rows; ++i) {
				tempArr[x1i].insert(tempArr[x1i].end(), tempMat.ptr<float>(i), tempMat.ptr<float>(i) + tempMat.cols * tempMat.channels());
			}
		}
	}
	
	xMat.create(cv::Size(0, tempArr[0].size()), CV_32FC1);

	for (unsigned int i = 0; i < tempArr.size(); ++i)
	{
		// Make a temporary cv::Mat row and add to NewSamples _without_ data copy
		cv::Mat Sample(1, tempArr[0].size(), CV_32FC1, tempArr[i].data());
		
		xMat.push_back(Sample);
	}

	Math::NeuralNetwork(xMat, yHatMat, wMat);
	Math::SoftMax(yHatMat, yHatMat);
}

void CNNMachine::BackPropagation()
{
	cv::Mat yLoss = -(yMat - yHatMat);
	cv::Mat wT = wMat.t();
	cv::Mat yLossW = yLoss*wT;
	std::cout << "yLossW size = " <<yLossW.size()<<std::endl;
	//벡터곱을 위해 yLossW를 pool2result 행렬 크기로 변환
	std::vector<std::vector<cv::Mat>> yLossWMats;
	//훈련 데이터 수(= USEDATA_NUM)만큼 반복
	for (int i = 0; i < poolresult2.size(); i++) {
		yLossWMats.push_back(std::vector<cv::Mat>());
		//poolresult2의 채널 수(= KERNEL2_NUM)만큼 반복
		for (int j = 0; j < poolresult2[0].size(); j++) {
			cv::Mat sample = yLossW.row(i).reshape(1, poolresult2[0].size()).row(j).reshape(1, poolresult2[0][0].rows);
			yLossWMats[i].push_back(sample);
		}
	}
	std::vector<std::vector<cv::Mat>> yLossWPoolUpMats;

	//변환된 yLossW를 풀링 필터로 Up-Sampleling 후 ReLu(Conv)행렬과 벡터곱
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		yLossWPoolUpMats.push_back(std::vector<cv::Mat>());
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			yLossWPoolUpMats[x1i].push_back(cv::Mat());

			//Pooling 함수 역방향 계산
			Math::GetMaxPoolingFilter(conv2ZeroPaddingMats[x1i][k2i], pool2BackpropFilters[x1i][k2i], poolresult2[x1i][k2i], poolSize, poolStride);
			
			//std::cout << pool2BackpropFilters[x1i][k2i].size() << std::endl;
			//std::cout << pool2BackpropFilters[x1i][k2i] << std::endl;
			//std::cout << yLossWMats[x1i][k2i].size() << std::endl;
			//std::cout << yLossWMats[x1i][k2i] << std::endl;

			//풀링 필터로 업샘플링
			Math::MaxPoolingBackprop(yLossWMats[x1i][k2i], yLossWPoolUpMats[x1i][k2i], pool2BackpropFilters[x1i][k2i], poolSize, poolStride);

			//std::cout << yLossWPoolUpMats[x1i][k2i] << std::endl;
			//std::cout << conv2Mats[x1i][k2i] << std::endl;
			
			//ReLu 함수 역방향 계산
			//(정방향 계산에서 이미 ReLU를 적용했으므로 생략)
			//Math::Relu(conv2Mats[x1i][k2i], conv2Mats[x1i][k2i]);
			
			//Up-Sampleling 결과 행렬과 ReLu(Conv)행렬을 벡터곱
			yLossWPoolUpMats[x1i][k2i] = yLossWPoolUpMats[x1i][k2i].mul(conv2Mats[x1i][k2i]);
			//std::cout << yLossWPoolUpMats[x1i][k2i] << std::endl;
		}
	}

	/*for (int i = 0; i < trainingMats.size(); i++) {
		for (int j = 0; j < KERNEL1_NUM; j++) {
			kernels1[i][j] = temp*wMat.t();
			std::cout << kernels1[i][j] << std::endl;
		}
	}*/

#pragma region 합성곱층1 가중치 행렬(커널1) 역방향 계산
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		//커널 1 채널 수만큼 반복
		for (int k1c = 0; k1c < kernels1.size(); k1c++) {
			//합성곱을 커널에 대해 미분 (커널 필터 얻기)
			std::cout << "합성곱1 입력 행렬 크기 : " << trainingMats[x1i].size() << ", 합성곱1 제로패딩 입력 행렬 크기 : " << x1ZeroPaddingMats[x1i].size() <<  ", 필터 행렬 크기 : " << conv1BackpropFilters[x1i][k1c].size() << std::endl;
			
			//Math::GetConvBackpropFilters(trainingMats[x1i], &conv1BackpropFilters[x1i][k1c], kernels1[0][0], kernel1Stride);

			//for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			//	cv::Mat kTemp;
			//	//Math::ConvKBackprop(yLossWMats[x1i][k2i], kTemp, conv2BackpropFilters[x1i][k1i], kernel2Stride);
			//	std::cout << kTemp << std::endl;
			//	kernels2[k1i][k2i] -= kTemp;
			//}
		}
	}
#pragma endregion

#pragma region 합성곱층2 가중치 행렬(커널2) 역방향 계산
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		//커널 2 채널 수만큼 반복
		for (int k2c = 0; k2c < kernels2.size(); k2c++) {
			//합성곱을 커널에 대해 미분 (커널 필터 얻기)
			std::cout << "합성곱2 입력 행렬 크기 : " << poolresult1[x1i][k2c].size() << "," << "필터 행렬 크기 : " << conv2BackpropFilters[x1i][k2c].size() << std::endl;
			Math::GetConvBackpropFilters(poolresult1[x1i][k2c], &conv2BackpropFilters[x1i][k2c], kernels2[0][0], kernel2Stride);
			
			//for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			//	cv::Mat kTemp;
			//	//Math::ConvKBackprop(yLossWMats[x1i][k2i], kTemp, conv2BackpropFilters[x1i][k1i], kernel2Stride);
			//	std::cout << kTemp << std::endl;
			//	kernels2[k1i][k2i] -= kTemp;
			//}
		}
	}

#pragma endregion

#pragma region 완전연결신경망층 가중치 행렬 역방향 계산
	//wMat -= (xMat.t() * (yLoss));
	//std::cout << wMat.size() << std::endl;
#pragma endregion
	
	
}

void CNNMachine::Training(int epoch, float learningRate, float l2)
{
	for (int i = 0; i < epoch; i++) {
		ForwardPropagation();
		BackPropagation();
	}
}
