#include "CNNMachine.h"

void CNNMachine::Training(int epoch, double learningRate, double l2)
{
	for (int i = 0; i < epoch; i++) {
		std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
		std::cout << i<<"번째 훈련" << std::endl;
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
		//std::cout << i<<"yMat : " << yMat << std::endl;
		//std::cout << "yHatMat : " << yHatMat << std::endl;
		std::cout << "코스트 : " << loss << std::endl;

		//아무키나 누르면 다음
		int key = cv::waitKey(0);
		if (key != -1) {
			std::cout << "역방향 계산" << std::endl;
			BackPropagation(learningRate);
			if (key == 13) //enter키
			{
				std::cout << "정방향 계산에서 얻은 yMat, yHatMat, yLoss로 역방향 계산 끝. " << std::endl;
				std::cout << "yMat(정답 행렬) : " << std::endl;
				std::cout << yMat << std::endl;
				std::cout << "yHatMat(가설 행렬) : " << std::endl;
				std::cout << yHatMat << std::endl;
				std::cout << "yLoss (= -(yMat - yHatMat)) : " << std::endl;
				std::cout << yLoss << std::endl;
			}
			continue;
		}
	}
}

void CNNMachine::Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec)
{
	for (int i = 0; i < trainingVec.size(); i++) {
		trainingMats.push_back(cv::Mat());
		//uchar형 행렬 요소를 double형 행렬 요소로 타입 캐스팅
		trainingVec[i].convertTo(trainingMats[i], CV_64FC1);
		trainingMats[i] /= 255;
	}

	poolStride = cv::Size(2, 2);
	poolSize = cv::Size(2, 2);
#pragma region 모든 가중치 행렬을 균등 분포로 랜덤 초기화, 커널 역방향 계산 필터 초기화
	cv::RNG gen(cv::getTickCount());

	//커널 1은 채널 한개(입력층 채널이 흑백 단일)
	kernels1.push_back(std::vector<cv::Mat>());
	for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
		kernels1[0].push_back(cv::Mat(cv::Size(3, 3), CV_64FC1));
		gen.fill(kernels1[0][k1i], cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(2));

		kernels2.push_back(std::vector<cv::Mat>());
		//커널 2는 채널이 커널 1의 개수
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
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

	w1Mat.create(cv::Size(10, wHeight * wWidth * KERNEL2_NUM), CV_64FC1);
	gen.fill(w1Mat, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(1));
	w2Mat.create(cv::Size(CLASSIFICATIONNUM, w1Mat.cols), CV_64FC1);
	gen.fill(w2Mat, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(1));

	
#pragma endregion
#pragma region 합성곱 결과, 합성곱 결과 제로 패딩, 풀링 결과, 풀링 결과 제로 패딩, 풀링 역방향 계산 필터 행렬 초기화
	for (int i = 0; i < trainingMats.size(); i++) {
		conv1Mats.push_back(std::vector<cv::Mat>());
		conv1Bias.push_back(std::vector<double>());
		conv2Mats.push_back(std::vector<cv::Mat>());
		conv2Bias.push_back(std::vector<double>());
		conv1ZeroPaddingMats.push_back(std::vector<cv::Mat>());
		conv2ZeroPaddingMats.push_back(std::vector<cv::Mat>());
		pool1result.push_back(std::vector<cv::Mat>());
		pool2result.push_back(std::vector<cv::Mat>());
		pool1resultZeroPadding.push_back(std::vector<cv::Mat>());

		pool1BackpropFilters.push_back(std::vector<cv::Mat>());
		pool2BackpropFilters.push_back(std::vector<cv::Mat>());
		for (int j = 0; j < KERNEL1_NUM; j++) {
			conv1Mats[i].push_back(cv::Mat_<double>());
			conv1Bias[i].push_back(0);
			conv1ZeroPaddingMats[i].push_back(cv::Mat_<double>());
			pool1result[i].push_back(cv::Mat_<double>());
			pool1resultZeroPadding[i].push_back(cv::Mat_<double>());
		
			pool1BackpropFilters[i].push_back(cv::Mat_<double>());
		}
		for (int j = 0; j < KERNEL2_NUM; j++) {
			conv2Mats[i].push_back(cv::Mat_<double>());
			conv2Bias[i].push_back(0);
			conv2ZeroPaddingMats[i].push_back(cv::Mat_<double>());
			pool2result[i].push_back(cv::Mat_<double>());
			
			pool2BackpropFilters[i].push_back(cv::Mat_<double>());
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
	yMat = cv::Mat::zeros(cv::Size(CLASSIFICATIONNUM, trainingMats.size()), CV_64FC1);
	for (int y = 0; y < labelVec.size(); y++) {
		//열과 맞는다면 true(1), 아니라면 false(0)를 저장
		yMat.at<double>(y, labelVec[y]) = 1;
	}

	yHatMat.create(cv::Size(CLASSIFICATIONNUM, trainingMats.size()), CV_64FC1);


	//lossAverage = 2305843009213693951;
}

void CNNMachine::ForwardPropagation()
{
	//합성곱층 결과 행렬을 완전연결신경망 입력으로 변환할 때 사용
	std::vector<std::vector<double>> tempArr;
	cv::Mat tempMat;

	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		tempArr.push_back(std::vector<double>());
		x1ZeroPaddingMats.push_back(cv::Mat_<double>());

		Math::CreateZeroPadding(trainingMats[x1i], x1ZeroPaddingMats[x1i], trainingMats[0].size(), kernels1[0][0].size(), kernel1Stride);
		//합성곱층 1
		for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
			Math::Convolution(x1ZeroPaddingMats[x1i], conv1Mats[x1i][k1i], trainingMats[0].size(), kernels1[0][k1i], kernel1Stride);
			Math::Relu(conv1Mats[x1i][k1i], conv1Mats[x1i][k1i]);
			Math::CreateZeroPadding(conv1Mats[x1i][k1i], conv1ZeroPaddingMats[x1i][k1i], pool1ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv1ZeroPaddingMats[x1i][k1i], pool1result[x1i][k1i], poolSize, poolStride);

			Math::CreateZeroPadding(pool1result[x1i][k1i], pool1resultZeroPadding[x1i][k1i], pool1result[0][0].size(), kernels2[0][0].size(), kernel2Stride);
		}
		//합성곱층 2
		/*합성곱층 1의 (행:데이터 수, 열:채널 수)의 이미지을 가진 pool1result행렬과
		합성곱층 2의 kernel2행렬을 행렬곱하듯 연결*/
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
				Math::Convolution(pool1resultZeroPadding[x1i][k1i], conv2Mats[x1i][k2i], pool1result[0][0].size(), kernels2[k1i][k2i], kernel2Stride);
			}
			Math::Relu(conv2Mats[x1i][k2i], conv2Mats[x1i][k2i]);
			Math::CreateZeroPadding(conv2Mats[x1i][k2i], conv2ZeroPaddingMats[x1i][k2i], pool2ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv2ZeroPaddingMats[x1i][k2i], pool2result[x1i][k2i], poolSize, poolStride);
		}
		//완전연결신경망 입력
		//4차원 pool2result를 2차원 행렬 xMat으로 변환
		//vec<vec<Mat>> to vec<vec<double>> 변환 : https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
		//vec<vec<double>> to Mat 변환 : https://stackoverflow.com/questions/18519647/opencv-convert-vector-of-vector-to-mat
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			tempMat = pool2result[x1i][k2i];
			/*imshow("Window", trainingMats[x1i]);
			cv::namedWindow("Windowaa", cv::WINDOW_NORMAL);
			imshow("Windowaa", tempMat);
			if (cv::waitKey(0) != -1)
				continue;*/
			for (int i = 0; i < tempMat.rows; ++i) {
				tempArr[x1i].insert(tempArr[x1i].end(), tempMat.ptr<double>(i), tempMat.ptr<double>(i) + tempMat.cols * tempMat.channels());
			}
		}
	}
	
	xMat.create(cv::Size(0, tempArr[0].size()), CV_64FC1);

	for (unsigned int i = 0; i < tempArr.size(); ++i)
	{
		// Make a temporary cv::Mat row and add to NewSamples _without_ data copy
		cv::Mat Sample(1, tempArr[0].size(), CV_64FC1, tempArr[i].data());
		xMat.push_back(Sample);
	}

	Math::NeuralNetwork(xMat, a1Mat, w1Mat);
	Math::Relu(a1Mat, a1Mat);

	Math::NeuralNetwork(a1Mat, yHatMat, w2Mat);
	Math::SoftMax(yHatMat, yHatMat);

	std::cout << "정방향 계산 커널1[0][0]\n" << kernels1[0][0] << std::endl;
	std::cout << "합성곱층 1 결과\n" << pool1result[0][0] << std::endl;
	std::cout << "정방향 계산 커널2[0][0]\n" << kernels2[0][0] << std::endl;
	std::cout << "합성곱층 2 결과\n" << pool2result[0][0] << std::endl;

}

void CNNMachine::BackPropagation(double learningRate)
{
	std::cout << std::fixed;
	yLoss = -(yMat - yHatMat); //손실함수를 SoftMax 함수 결과에 대해 미분한 값
	w2T = w2Mat.t();
	yLossW2 = yLoss*w2T; //손실 함수를 완전연결층2 입력(ReLu(aMat))에 대해 미분한 값

	//Relu(a1Mat)과 벡터곱
	std::cout << "ReLu 벡터곱 전\n" << yLossW2 << std::endl;

	yLossW2Relu3 = yLossW2.mul(a1Mat); //손실함수를 완전연결층1 결과에 대해 미분한 값
	//Math::Relu(yLossW2Relu3, yLossW2Relu3);
	
	std::cout << "ReLu 벡터곱 후\n" << yLossW2Relu3	<< std::endl;

	yLossW2Relu3W1 = yLossW2Relu3 * w1Mat.t();//손실 함수를 완전연결층1 입력에 대해 미분한 값

	std::vector<std::vector<cv::Mat>> yLossW2Relu3W1Temp; //yLossW2Relu3W1를 풀링2결과행렬의 크기로 차원 변환
	std::vector<std::vector<cv::Mat>> yLossW2Relu3W1UpRelu2; //손실 함수를 합성곱2 결과에 대해 미분한 값 (Up은 Up-Sampleling(풀링함수의 미분))
	std::vector<std::vector<cv::Mat>> yLossW2Relu3W1UpRelu2P1UpRelu; //손실 함수를 합성곱1 결과에 대해 미분한 값

	//벡터곱을 위해 yLossW2Relu3W1를 풀링2 결과 행렬 크기로 변환
	for (int i = 0; i < trainingMats.size(); i++) {
		yLossW2Relu3W1Temp.push_back(std::vector<cv::Mat>());
		for (int j = 0; j < KERNEL2_NUM; j++) {
			cv::Mat sample = yLossW2Relu3W1.row(i).reshape(1, pool2result[0].size()).row(j).reshape(1, pool2result[0][0].rows);
			yLossW2Relu3W1Temp[i].push_back(sample);
		}
	}

	//차원 변환된 yLossW2Relu3W1를 풀링2 필터로 Up-Sampleling 후 Relu(Conv2)행렬과 벡터곱
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		yLossW2Relu3W1UpRelu2.push_back(std::vector<cv::Mat>());
		for (int k2n = 0; k2n < KERNEL2_NUM; k2n++) {
			yLossW2Relu3W1UpRelu2[x1i].push_back(cv::Mat());
			//Pooling 함수 역방향 계산으로 풀링 필터 할당
			Math::GetMaxPoolingFilter(conv2ZeroPaddingMats[x1i][k2n], pool2BackpropFilters[x1i][k2n], pool2result[x1i][k2n], poolSize, poolStride);
			//풀링 필터로 업샘플링
			Math::MaxPoolingBackprop(yLossW2Relu3W1Temp[x1i][k2n], yLossW2Relu3W1UpRelu2[x1i][k2n], pool2BackpropFilters[x1i][k2n], poolSize, poolStride);

			//Relu 함수 역방향 계산
			//Up-Sampleling 결과 행렬과 Relu(Conv2)행렬을 벡터곱
			yLossW2Relu3W1UpRelu2[x1i][k2n] = yLossW2Relu3W1UpRelu2[x1i][k2n].mul(conv2Mats[x1i][k2n]);
			//Math::Relu(yLossW2Relu3W1UpRelu2[x1i][k2n], yLossW2Relu3W1UpRelu2[x1i][k2n]);
			
			//std::cout << "Conv2Mats\n" << conv2Mats[x1i][k2n] << std::endl;
			//std::cout << "ReLu Conv2와의 벡터곱 후\n" << yLossW2Relu3W1UpRelu2[x1i][k2n] << std::endl;
		}
	}
	
	//커널2 역방향 계산을 위한 합성곱2 필터 계산
	Math::GetConvBackpropFilters(pool1result[0][0], &conv2BackpropFilters, kernels2[0][0], kernel2Stride);
	//커널1 역방향 계산을 위한 합성곱1 필터 계산
	Math::GetConvBackpropFilters(trainingMats[0], &conv1BackpropFilters, kernels1[0][0], kernel1Stride);

	//yLossW2Relu3W1UpRelu2행렬과 합성곱2 함수의 커널2에 대한 미분 행렬을 벡터곱하고, 풀링1 필터로 Up-Sampleling 후 Relu(Conv1)행렬과 벡터곱
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		yLossW2Relu3W1UpRelu2P1UpRelu.push_back(std::vector<cv::Mat>());
		//커널 1 개수만큼 반복
		for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
			yLossW2Relu3W1UpRelu2P1UpRelu[x1i].push_back(cv::Mat());

			cv::Mat yLossW2Relu3W1UpRelu2P1 = cv::Mat(yLossW2Relu3W1UpRelu2[x1i][0].size(), CV_64FC1);
			yLossW2Relu3W1UpRelu2P1.setTo(0);

			//커널 2 개수만큼 반복
			for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
				cv::Mat k2Temp;
				//yLossW2Relu3W1UpRelu2행렬과 합성곱2 함수의 커널2에 대한 미분 행렬을 벡터곱
				Math::ConvXBackprop(yLossW2Relu3W1UpRelu2[x1i][k2n], kernels2[k1n][k2n], k2Temp, conv2BackpropFilters, kernel1Stride, learningRate);
				yLossW2Relu3W1UpRelu2P1 += k2Temp;
			}
			//평균치 계산
			yLossW2Relu3W1UpRelu2P1 /= kernels2[0].size();
			//Pooling 함수 역방향 계산으로 풀링 필터 정의
			Math::GetMaxPoolingFilter(conv1ZeroPaddingMats[x1i][k1n], pool1BackpropFilters[x1i][k1n], pool1result[x1i][k1n], poolSize, poolStride);
			//풀링 필터로 업샘플링
			Math::MaxPoolingBackprop(yLossW2Relu3W1UpRelu2P1, yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n], pool1BackpropFilters[x1i][k1n], poolSize, poolStride);

			//Relu 함수 역방향 계산
			//(정방향 계산에서 이미 ReLU를 적용했으므로 생략)
			//Math::Relu(conv1Mats[x1i][k1n], conv1Mats[x1i][k1n]);
			yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n] = yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n].mul(conv1Mats[x1i][k1n]);
			//Math::Relu(yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n], yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n]);
		}
	}
#pragma region 합성곱층1 가중치 행렬(커널1) 역방향 계산
	//std::cout << "\n커널 1 역방향 계산 " << std::endl;
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		for (int k1c = 0; k1c < kernels1.size(); k1c++) {
			for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
				cv::Mat newKernel;
				Math::ConvKBackprop(-yLossW2Relu3W1UpRelu2P1UpRelu[x1i][k1n], x1ZeroPaddingMats[x1i], kernels1[k1c][k1n], newKernel, conv1BackpropFilters, kernel1Stride, learningRate);
				newKernel.copyTo(kernels1[k1c][k1n]);
				//std::cout << "커널 역방향 행렬 업데이트 후 : " << kernels1[k1c][k1n] << std::endl;
			}
		}
	}
#pragma endregion

#pragma region 합성곱층2 가중치 행렬(커널2) 역방향 계산
	//std::cout << "\n커널 2 역방향 계산 " << std::endl;
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		for (int k2c = 0; k2c < kernels2.size(); k2c++) {
			for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
				cv::Mat newKernel;

				Math::ConvKBackprop(-yLossW2Relu3W1UpRelu2[x1i][k2n], pool1resultZeroPadding[x1i][k2c], kernels2[k2c][k2n], newKernel, conv2BackpropFilters, kernel2Stride, learningRate);
				newKernel.copyTo(kernels2[k2c][k2n]);
				//std::cout << "커널 역방향 행렬 업데이트 후 : " << kernels2[k2c][k2n] << std::endl;
			}
		}
	}

#pragma endregion

#pragma region 완전연결신경망층 가중치 행렬 역방향 계산
	w1Mat -= learningRate * xMat.t()*(yLossW2Relu3) / w2Mat.size().width;
	w2Mat -= learningRate *(a1Mat.t() * (yLoss)) / yLoss.size().width;
	//std::cout << w1Mat << std::endl;
#pragma endregion
	
	
}


