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
	gen.fill(wMat, cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(10));

	
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
	//합성곱층 결과 행렬을 완전연결신경망 입력으로 변환할 때 사용
	std::vector<std::vector<float>> tempArr;
	cv::Mat tempMat;

	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		tempArr.push_back(std::vector<float>());
		x1ZeroPaddingMats.push_back(cv::Mat_<float>());

		Math::CreateZeroPadding(trainingMats[x1i], x1ZeroPaddingMats[x1i], trainingMats[0].size(), kernels1[0][0].size(), kernel1Stride);
		//합성곱층 1
		for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
			Math::Convolution(x1ZeroPaddingMats[x1i], conv1Mats[x1i][k1i], trainingMats[0].size(), kernels1[0][k1i], kernel1Stride);
			Math::Relu(conv1Mats[x1i][k1i], conv1Mats[x1i][k1i]);
			Math::CreateZeroPadding(conv1Mats[x1i][k1i], conv1ZeroPaddingMats[x1i][k1i], pool1ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv1ZeroPaddingMats[x1i][k1i], poolresult1[x1i][k1i], poolSize, poolStride);

			Math::CreateZeroPadding(poolresult1[x1i][k1i], poolresult1ZeroPadding[x1i][k1i], poolresult1[0][0].size(), kernels2[0][0].size(), kernel2Stride);
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

void CNNMachine::BackPropagation(float learningRate)
{
	cv::Mat yLoss = -(yMat - yHatMat); //손실함수를 SoftMax 함수 결과에 대해 미분한 값
	cv::Mat wT = wMat.t();
	cv::Mat yLossW = yLoss*wT; //손실 함수를 완전연결층 W에 대해 미분한 값
	std::vector<std::vector<cv::Mat>> yLossWTemp; //yLossW를 풀링2결과행렬의 크기로 차원 변환

	std::vector<std::vector<cv::Mat>> yLossWUpRelu2; //손실 함수를 합성곱2 결과에 대해 미분한 값 (Up은 Up-Sampleling(풀링함수의 미분) 약자)
	std::vector<std::vector<cv::Mat>> yLossWUpRelu2P1UpRelu; //손실 함수를 합성곱1 결과에 대해 미분한 값

	//벡터곱을 위해 yLossW를 풀링2 결과 행렬 크기로 변환
	for (int i = 0; i < trainingMats.size(); i++) {
		yLossWTemp.push_back(std::vector<cv::Mat>());
		for (int j = 0; j < KERNEL2_NUM; j++) {
			cv::Mat sample = yLossW.row(i).reshape(1, poolresult2[0].size()).row(j).reshape(1, poolresult2[0][0].rows);
			yLossWTemp[i].push_back(sample);
		}
	}
	std::cout << yMat << std::endl;
	std::cout << yHatMat << std::endl;
	std::cout << yLoss << std::endl;
	std::cout << wT << std::endl;
	std::cout << yLossWTemp[0][0] << std::endl;

	//차원 변환된 yLossW를 풀링 필터로 Up-Sampleling 후 Relu(Conv2)행렬과 벡터곱
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		yLossWUpRelu2.push_back(std::vector<cv::Mat>());
		for (int k2n = 0; k2n < KERNEL2_NUM; k2n++) {
			yLossWUpRelu2[x1i].push_back(cv::Mat());
			//Pooling 함수 역방향 계산으로 풀링 필터 할당
			Math::GetMaxPoolingFilter(conv2ZeroPaddingMats[x1i][k2n], pool2BackpropFilters[x1i][k2n], poolresult2[x1i][k2n], poolSize, poolStride);
			//풀링 필터로 업샘플링
			Math::MaxPoolingBackprop(yLossWTemp[x1i][k2n], yLossWUpRelu2[x1i][k2n], pool2BackpropFilters[x1i][k2n], poolSize, poolStride);

			//Relu 함수 역방향 계산
			//(정방향 계산에서 이미 ReLU를 적용했으므로 생략)
			//Math::Relu(conv2Mats[x1i][k2n], conv2Mats[x1i][k2n]);

			std::cout << "ReLu Conv2와의 곱 전\n" << yLossWUpRelu2[x1i][k2n] << std::endl;
			//Up-Sampleling 결과 행렬과 Relu(Conv2)행렬을 벡터곱
			//yLossWUpRelu2[x1i][k2n] = yLossWUpRelu2[x1i][k2n].mul(conv2Mats[x1i][k2n]);
			//Math::Relu(yLossWUpRelu2[x1i][k2n], yLossWUpRelu2[x1i][k2n]);
			//std::cout << "ReLu Conv2와의 곱 후\n" << yLossWUpRelu2[x1i][k2n] << std::endl;
		}
	}
	
	//커널2 역방향 계산을 위한 합성곱2 필터 계산
	Math::GetConvBackpropFilters(poolresult1[0][0], &conv2BackpropFilters, kernels2[0][0], kernel2Stride);
	//커널1 역방향 계산을 위한 합성곱1 필터 계산
	Math::GetConvBackpropFilters(trainingMats[0], &conv1BackpropFilters, kernels1[0][0], kernel1Stride);
//
//#pragma region 합성곱층1 가중치 행렬(커널1) 역방향 계산
//	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
//		//yLossWUpRelu2P1.push_back(std::vector<cv::Mat>());
//		yLossWUpRelu2P1UpRelu.push_back(std::vector<cv::Mat>());
//		for (int k1c = 0; k1c < kernels1.size(); k1c++) {
//			for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
//				
//			}
//		}
//		//커널 2 채널 수만큼 반복
//		for (int k2c = 0; k2c < kernels2.size(); k2c++) {
//			//커널 2 개수만큼 반복
//			for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
//				yLossWUpRelu2P1UpRelu[x1i].push_back(cv::Mat());
//				//yLossWUpRelu2P1[x1i].push_back(cv::Mat()); 
//				cv::Mat yLossWUpRelu2P1;
//				Math::ConvXBackprop(yLossWUpRelu2[x1i][k2n], kernels2[k2c][k2n], yLossWUpRelu2P1, conv2BackpropFilters, kernel2Stride);
//				
//				//Pooling 함수 역방향 계산
//				Math::GetMaxPoolingFilter(conv1ZeroPaddingMats[x1i][k2c], pool1BackpropFilters[x1i][k2c], poolresult1[x1i][k2c], poolSize, poolStride);
//				//풀링 필터로 업샘플링
//				Math::MaxPoolingBackprop(yLossWUpRelu2P1, yLossWUpRelu2[x1i][k2n], pool2BackpropFilters[x1i][k2n], poolSize, poolStride);
//				//Relu 함수 역방향 계산
//				//(정방향 계산에서 이미 ReLU를 적용했으므로 생략)
//				//Math::Relu(conv1Mats[x1i][k2c], conv2Mats[x1i][k2c]);
//				
//			}
//		}
//
//		//커널 1 채널 수만큼 반복
//		for (int k1c = 0; k1c < kernels1.size(); k1c++) {
//			//합성곱을 커널에 대해 미분 (커널 필터 얻기)
//			std::cout << "합성곱1 입력 행렬 크기 : " << trainingMats[x1i].size() << ", 합성곱1 제로패딩 입력 행렬 크기 : " << x1ZeroPaddingMats[x1i].size() <<  ", 필터 행렬 크기 : " << conv1BackpropFilters.size() << std::endl;
//			
//
//			//for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
//			//	cv::Mat kTemp;
//			//	//Math::ConvKBackprop(yLossWTemp[x1i][k2i], kTemp, conv2BackpropFilters[x1i][k1i], kernel2Stride);
//			//	std::cout << kTemp << std::endl;
//			//	kernels2[k1i][k2i] -= kTemp;
//			//}
//		}
//	}
//#pragma endregion

#pragma region 합성곱층2 가중치 행렬(커널2) 역방향 계산
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		//커널 2 채널 수만큼 반복
		for (int k2c = 0; k2c < kernels2.size(); k2c++) {
			//합성곱을 커널에 대해 미분 (커널 필터 얻기)
			//std::cout << "합성곱2 입력 행렬 크기 : " << poolresult1[x1i][k2c].size() << "," << "필터 행렬 크기 : " << conv2BackpropFilters.size() << std::endl;
			//std::cout << "역방향 행렬 크기 : " << yLossWUpRelu2[x1i][0].size() << std::endl;
			for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
				//std::cout << "역방향 행렬 업데이트 전 : " << kernels2[k2c][k2n] << std::endl;

				cv::Mat newInput = yLossWUpRelu2[x1i][k2n].mul(poolresult1[x1i][k2c]);
				cv::Mat newKernel;
				std::cout << yLossWUpRelu2[x1i][k2n] << std::endl;
				std::cout << poolresult1ZeroPadding[0][0] << std::endl;

				//std::cout << newInput << std::endl;

				Math::ConvKBackprop(-newInput,kernels2[k2c][k2n], newKernel, conv2BackpropFilters, kernel2Stride);
				newKernel.copyTo(kernels2[k2c][k2n]);
				//std::cout << "역방향 행렬 업데이트 후 : " << kernels2[k2c][k2n] << std::endl;
			}
		}
	}

#pragma endregion

#pragma region 완전연결신경망층 가중치 행렬 역방향 계산
	wMat -= learningRate *(xMat.t() * (yLoss));
	//std::cout << wMat << std::endl;
#pragma endregion
	
	
}

void CNNMachine::Training(int epoch, float learningRate, float l2)
{
	for (int i = 0; i < epoch; i++) {
		ForwardPropagation();
		cost = 0;
		for (int y = 0; y < yMat.rows; y++) {
			for (int x = 0; x < yMat.cols; x++) {
				//log(0) 음의 무한대 예외처리로 0 대신 0에 가까운 수 사용
				cost += yMat.at<float>(y, x) * log((yHatMat.at<float>(y, x) == 0) ? 0.00000000001 : yHatMat.at<float>(y, x));
			}
		}
		cost /= -yMat.rows;
		//std::cout << i<<"yMat : " << yMat << std::endl;
		//std::cout << "yHatMat : " << yHatMat << std::endl;
		std::cout << "코스트 : " << cost << std::endl;

		//아무키나 누르면 다음
		if (cv::waitKey(0) != -1) {
			BackPropagation(learningRate);
			continue;
		}
	}
}
