#include "CNNMachine.h"

void CNNMachine::Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec)
{
	for (int i = 0; i < trainingVec.size(); i++) {
		trainingMats.push_back(cv::Mat());
		trainingVec[i].convertTo(trainingMats[i], CV_32FC1);
	}
	//std::cout << trainingMats[0].type() << std::endl;
	//y

	poolStride = cv::Size(2, 2);
	poolSize = cv::Size(2, 2);
#pragma region 모든 가중치 행렬을 균등 분포로 랜덤 초기화
	cv::RNG gen(cv::getTickCount());
	//커널 1은 채널 한개
	kernels1.push_back(std::vector<cv::Mat>());
	for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
		kernels1[0].push_back(cv::Mat(cv::Size(3, 3), CV_32FC1));
		gen.fill(kernels1[0][k1i], cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(1));

		kernels2.push_back(std::vector<cv::Mat>());
		//커널 2는 채널이 커널 1의 갯수
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			kernels2[k1i].push_back(cv::Mat(cv::Size(3, 3), CV_32FC1));
			gen.fill(kernels2[k1i][k2i], cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(1));
		}
	}
	
	kernel1Stride = cv::Size(1,1);
	kernel2Stride = cv::Size(1,1);
	
	//합성곱은 세임 패딩으로 진행하므로 풀링층 2개에서의 축소만 계산
	int wHeight = (trainingMats[0].rows - poolSize.height) / poolStride.height + 1;
		wHeight = (wHeight - poolSize.height) / poolStride.height + 1;
	int wWidth = (trainingMats[0].cols - poolSize.width) / poolStride.width + 1;
		wWidth = (wWidth - poolSize.width) / poolStride.width + 1;

	wMat.create(cv::Size(CLASSIFICATIONNUM, wHeight * wWidth * KERNEL2_NUM), CV_32FC1);
	gen.fill(wMat, cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(1));
#pragma endregion
	for (int i = 0; i < trainingMats.size(); i++) {
		conv1Mats.push_back(std::vector<cv::Mat>());
		conv2Mats.push_back(std::vector<cv::Mat>());
		conv1ZeroPaddingMats.push_back(std::vector<cv::Mat>());
		conv1PoolFilters.push_back(std::vector<cv::Mat>());
		conv2ZeroPaddingMats.push_back(std::vector<cv::Mat>());
		conv2PoolFilters.push_back(std::vector<cv::Mat>());
		poolresult1.push_back(std::vector<cv::Mat>());
		poolresult2.push_back(std::vector<cv::Mat>());
		poolresult1ZeroPadding.push_back(std::vector<cv::Mat>());
		for (int j = 0; j < KERNEL1_NUM; j++) {
			conv1Mats[i].push_back(cv::Mat_<float>());
			conv1ZeroPaddingMats[i].push_back(cv::Mat_<float>());
			conv1PoolFilters[i].push_back(cv::Mat_<float>());
			poolresult1[i].push_back(cv::Mat_<float>());
		}
		for (int j = 0; j < KERNEL2_NUM; j++) {
			//poolresult1제로패딩은 커널 2와 연결하기에 수를 같게함
			poolresult1ZeroPadding[i].push_back(cv::Mat_<float>());
			conv2Mats[i].push_back(cv::Mat_<float>());
			conv2ZeroPaddingMats[i].push_back(cv::Mat_<float>());
			conv2PoolFilters[i].push_back(cv::Mat_<float>());
			poolresult2[i].push_back(cv::Mat_<float>());
		}
	}

	//정답 데이터를 벡터로 변환한다.
	yMat = cv::Mat::zeros(cv::Size(CLASSIFICATIONNUM, trainingMats.size()), CV_32FC1);
	/*std::cout << "정답 행렬 : \n" << CLASSIFICATIONNUM <<","<< trainingMats.size() << std::endl;
	std::cout << yMat.size() << std::endl;
	std::cout << "정답 행렬 : \n" << yMat << std::endl;*/
	for (int y = 0; y < labelVec.size(); y++) {
		//열과 맞는다면 true(1), 아니라면 false(0)를 저장
		yMat.at<float>(y, labelVec[y]) = 1;
	}
	//std::cout << "정답 행렬 : \n" << yMat << std::endl;

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
		if (x1i % 10 == 0)
			std::cout << x1i << std::endl;
		tempArr.push_back(std::vector<float>());
		x1ZeroPaddingMats.push_back(cv::Mat_<float>());

		Math::CreateZeroPadding(trainingMats[x1i], x1ZeroPaddingMats[x1i], trainingMatrixSize, k1MatrixSize, kernel1Stride);

		//합성곱층 1
		for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
			Math::Convolution(x1ZeroPaddingMats[x1i], conv1Mats[x1i][k1i], trainingMatrixSize, kernels1[0][k1i], kernel1Stride);
			//std::cout << k1i<<"합성곱전\n"<< conv1Mats[x1i][k1i] << std::endl;
			Math::Relu(conv1Mats[x1i][k1i], conv1Mats[x1i][k1i]);
			//std::cout << k1i << "후\n" << conv1Mats[x1i][k1i] << std::endl;
			Math::CreateZeroPadding(conv1Mats[x1i][k1i], conv1ZeroPaddingMats[x1i][k1i], pool1ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv1ZeroPaddingMats[x1i][k1i], poolresult1[x1i][k1i], poolSize, poolStride);

			//합성곱층 2
			/*합성곱층 1의 (행:데이터 수, 열:채널 수)의 이미지을 가진 poolresult1행렬과
			합성곱층 2의 kernel2행렬을 행렬곱하듯 연결*/
			for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
				Math::CreateZeroPadding(poolresult1[x1i][k1i], poolresult1ZeroPadding[x1i][k1i], poolresult1[0][0].size(), k2MatrixSize, kernel2Stride);
				Math::Convolution(poolresult1ZeroPadding[x1i][k1i], conv2Mats[x1i][k2i], poolresult1[0][0].size(), kernels2[k1i][k2i], kernel2Stride);
				Math::Relu(conv2Mats[x1i][k2i], conv2Mats[x1i][k2i]);
				Math::CreateZeroPadding(conv2Mats[x1i][k2i], conv2ZeroPaddingMats[x1i][k2i], pool2ResultSize, poolSize, poolStride);
				Math::MaxPooling(conv2ZeroPaddingMats[x1i][k2i], poolresult2[x1i][k2i], poolSize, poolStride);

				Math::MaxPoolingReverse(conv2Mats[x1i][k2i], conv2PoolFilters[x1i][k2i], poolresult2[x1i][k2i]);
				std::cout << conv2Mats[x1i][k2i] << std::endl;
				std::cout << poolresult2[x1i][k2i] << std::endl;
				std::cout << conv2PoolFilters[x1i][k2i] << std::endl;
			}
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
	//std::cout << tempMat.size() << std::endl;
	//std::cout << poolresult2.size() << ",," << poolresult2[0].size() << poolresult2[0][0].size() << std::endl;
	//std::cout << tempArr.size() << ",,"<<tempArr[0].size() << std::endl;
	//std::cout << xMat.type() << std::endl;
	//std::cout << xMat.size() << std::endl;
	//std::cout << xMat << std::endl;
	//std::cout << wMat.type() << std::endl;
	//std::cout << wMat.size() << std::endl;
	//std::cout << wMat << std::endl;
	//std::cout << yHatMat.type() << std::endl;
	//std::cout << yHatMat.size() << std::endl;
	
	Math::NeuralNetwork(xMat, yHatMat, wMat);
	//std::cout << yHatMat << std::endl;
	Math::SoftMax(yHatMat, yHatMat);
	
	//std::cout << yHatMat.type() << std::endl;
	//std::cout << yHatMat.size() << std::endl;
	//std::cout << yHatMat << std::endl;
}

void CNNMachine::BackPropagation()
{
	cv::Mat temp = yMat - yHatMat;
	
	wMat -= -(xMat.t() * (temp));
	std::cout << wMat.size() << std::endl;

	//xMat 크기를 가진 행렬을 역풀링 함수로 합성곱 연산 결과 형태로 변환
	std::vector<std::vector<cv::Mat>> temp2
		//= temp * wMat.t()
		;
		
	/*for (int i = 0; i < trainingMats.size(); i++) {
		for (int j = 0; j < KERNEL2_NUM; j++) {
			kernels2[i][j] = 
			std::cout << kernels2[i][j] << std::endl;
		}
	}*/
	
	/*for (int i = 0; i < trainingMats.size(); i++) {
		for (int j = 0; j < KERNEL1_NUM; j++) {
			kernels1[i][j] = temp*wMat.t();
			std::cout << kernels1[i][j] << std::endl;
		}
	}*/
}

void CNNMachine::Training(int epoch, float learningRate, float l2)
{
	for (int i = 0; i < epoch; i++) {
		ForwardPropagation();
		BackPropagation();
	}
}
