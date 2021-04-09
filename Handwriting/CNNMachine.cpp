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
#pragma region ��� ����ġ ����� �յ� ������ ���� �ʱ�ȭ, Ŀ�� ������ ��� ���� �ʱ�ȭ
	cv::RNG gen(cv::getTickCount());

	//Ŀ�� 1�� ä�� �Ѱ�(�Է��� ä���� ��� ����)
	kernels1.push_back(std::vector<cv::Mat>());
	for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
		kernels1[0].push_back(cv::Mat(cv::Size(3, 3), CV_32FC1));
		gen.fill(kernels1[0][k1i], cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(1));

		kernels2.push_back(std::vector<cv::Mat>());
		//Ŀ�� 2�� ä���� Ŀ�� 1�� ����
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			kernels2[k1i].push_back(cv::Mat(cv::Size(3, 3), CV_32FC1));
			gen.fill(kernels2[k1i][k2i], cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(1));
		}
		
		
	}
	kernel1Stride = cv::Size(1, 1);
	kernel2Stride = cv::Size(1, 1);
	
	//�ռ����� ���� �е����� �����ϹǷ� Ǯ���� 2�������� ��Ҹ� ���
	int wHeight = (trainingMats[0].rows - poolSize.height) / poolStride.height + 1;
		wHeight = (wHeight - poolSize.height) / poolStride.height + 1;
	int wWidth = (trainingMats[0].cols - poolSize.width) / poolStride.width + 1;
		wWidth = (wWidth - poolSize.width) / poolStride.width + 1;

	wMat.create(cv::Size(CLASSIFICATIONNUM, wHeight * wWidth * KERNEL2_NUM), CV_32FC1);
	gen.fill(wMat, cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(1));

	//1��° �ռ��� Ŀ���� ������ ��� ���� �ʱ�ȭ
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		conv1Filters.push_back(std::vector<std::vector<std::vector<std::vector<float>>>>());
		//Ŀ�� 1�� ä�� �Ѱ�(�Է��� ä���� ��� ����)
		for (int k1i = 0; k1i < kernels1.size(); k1i++) {
			conv1Filters[x1i].push_back(std::vector<std::vector<std::vector<float>>>());

			int kf1Num = kernels1[0][0].rows * kernels1[0][0].cols;
			for (int kf1Y = 0; kf1Y < kernels1[0][0].rows; kf1Y++) {
				conv1Filters[x1i][k1i].push_back(std::vector<std::vector<float>>());
				for (int kf1X = 0; kf1X < kernels1[0][0].cols; kf1X++) {
					conv1Filters[x1i][k1i][kf1Y].push_back(std::vector<float>());
					//K�� �������� ��� ����ŭ �޸� �Ҵ�
					for (int kf1N = 0; kf1N < kf1Num; kf1N++) {
						conv1Filters[x1i][0][kf1Y][kf1X].push_back(0);
					}
				}
			}
		}
	}
	//2��° �ռ��� Ŀ���� ������ ��� ���� �ʱ�ȭ
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		conv2Filters.push_back(std::vector<std::vector<std::vector<std::vector<float>>>>());
		//Ŀ�� 2�� ä�� ������ Ŀ��1�� ����
		for (int k1i = 0; k1i < kernels1[0].size(); k1i++) {
			conv2Filters[x1i].push_back(std::vector<std::vector<std::vector<float>>>());

			int kf2Num = kernels2[0][0].rows * kernels2[0][0].cols;
			for (int kf2Y = 0; kf2Y < kernels2[0][0].rows; kf2Y++) {
				conv2Filters[x1i][k1i].push_back(std::vector<std::vector<float>>());
				for (int kf2X = 0; kf2X < kernels2[0][0].cols; kf2X++) {
					conv2Filters[x1i][k1i][kf2Y].push_back(std::vector<float>());
					//K�� �������� ��� ����ŭ �޸� �Ҵ�
					for (int kf2N = 0; kf2N < kf2Num; kf2N++) {
						conv2Filters[x1i][k1i][kf2Y][kf2X].push_back(0);
					}
				}
			}
		}
	}
#pragma endregion
#pragma region �ռ��� ���, �ռ��� ��� ���� �е�, Ǯ�� ���, Ǯ�� ��� ���� �е�, Ǯ�� ������ ��� ���� ��� �ʱ�ȭ
	for (int i = 0; i < trainingMats.size(); i++) {
		conv1Mats.push_back(std::vector<cv::Mat>());
		conv2Mats.push_back(std::vector<cv::Mat>());
		conv1ZeroPaddingMats.push_back(std::vector<cv::Mat>());
		conv2ZeroPaddingMats.push_back(std::vector<cv::Mat>());
		poolresult1.push_back(std::vector<cv::Mat>());
		poolresult2.push_back(std::vector<cv::Mat>());
		poolresult1ZeroPadding.push_back(std::vector<cv::Mat>());

		pool1Filters.push_back(std::vector<cv::Mat>());
		pool2Filters.push_back(std::vector<cv::Mat>());
		for (int j = 0; j < KERNEL1_NUM; j++) {
			conv1Mats[i].push_back(cv::Mat_<float>());
			conv1ZeroPaddingMats[i].push_back(cv::Mat_<float>());
			poolresult1[i].push_back(cv::Mat_<float>());
			poolresult1ZeroPadding[i].push_back(cv::Mat_<float>());
		
			pool1Filters[i].push_back(cv::Mat_<float>());
		}
		for (int j = 0; j < KERNEL2_NUM; j++) {
			conv2Mats[i].push_back(cv::Mat_<float>());
			conv2ZeroPaddingMats[i].push_back(cv::Mat_<float>());
			poolresult2[i].push_back(cv::Mat_<float>());
			
			pool2Filters[i].push_back(cv::Mat_<float>());
		}
	}
#pragma endregion


	//���� �����͸� ���ͷ� ��ȯ�Ѵ�.
	yMat = cv::Mat::zeros(cv::Size(CLASSIFICATIONNUM, trainingMats.size()), CV_32FC1);
	/*std::cout << "���� ��� : \n" << CLASSIFICATIONNUM <<","<< trainingMats.size() << std::endl;
	std::cout << yMat.size() << std::endl;
	std::cout << "���� ��� : \n" << yMat << std::endl;*/
	for (int y = 0; y < labelVec.size(); y++) {
		//���� �´´ٸ� true(1), �ƴ϶�� false(0)�� ����
		yMat.at<float>(y, labelVec[y]) = 1;
	}
	//std::cout << "���� ��� : \n" << yMat << std::endl;

	yHatMat.create(cv::Size(CLASSIFICATIONNUM, trainingMats.size()), CV_32FC1);


	//lossAverage = 2305843009213693951;
}

void CNNMachine::ForwardPropagation()
{
	//��� ������
	cv::Size trainingMatrixSize = trainingMats[0].size();
	cv::Size k1MatrixSize = kernels1[0][0].size();
	cv::Size k2MatrixSize = kernels2[0][0].size();
	//�ռ��� �� ���� �е��� ����ϹǷ� Ǯ�� ����� ���
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

		//�ռ����� 1
		for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
			Math::Convolution(x1ZeroPaddingMats[x1i], conv1Mats[x1i][k1i], trainingMatrixSize, kernels1[0][k1i], kernel1Stride);
			//std::cout << k1i<<"�ռ�����\n"<< conv1Mats[x1i][k1i] << std::endl;
			Math::Relu(conv1Mats[x1i][k1i], conv1Mats[x1i][k1i]);
			//std::cout << k1i << "��\n" << conv1Mats[x1i][k1i] << std::endl;
			Math::CreateZeroPadding(conv1Mats[x1i][k1i], conv1ZeroPaddingMats[x1i][k1i], pool1ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv1ZeroPaddingMats[x1i][k1i], poolresult1[x1i][k1i], poolSize, poolStride);

			Math::CreateZeroPadding(poolresult1[x1i][k1i], poolresult1ZeroPadding[x1i][k1i], poolresult1[0][0].size(), k2MatrixSize, kernel2Stride);
		}
		//�ռ����� 2
		/*�ռ����� 1�� (��:������ ��, ��:ä�� ��)�� �̹����� ���� poolresult1��İ�
		�ռ����� 2�� kernel2����� ��İ��ϵ� ����*/
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			//std::cout << poolresult1[x1i][k1i] << std::endl;
			//std::cout << poolresult1ZeroPadding[x1i][k1i] << std::endl;
			for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
				Math::Convolution(poolresult1ZeroPadding[x1i][k1i], conv2Mats[x1i][k2i], poolresult1[0][0].size(), kernels2[k1i][k2i], kernel2Stride);
			}
			Math::Relu(conv2Mats[x1i][k2i], conv2Mats[x1i][k2i]);
			Math::CreateZeroPadding(conv2Mats[x1i][k2i], conv2ZeroPaddingMats[x1i][k2i], pool2ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv2ZeroPaddingMats[x1i][k2i], poolresult2[x1i][k2i], poolSize, poolStride);
			Math::GetMaxPoolingFilter(conv2Mats[x1i][k2i], pool2Filters[x1i][k2i], poolresult2[x1i][k2i], poolSize, poolStride);
			//std::cout << "���� Ŀ��1��"<< k1i << "Ŀ��2��"<< k2i << std::endl;
			//
			//std::cout << conv2Mats[x1i][k2i] << std::endl;
			//std::cout << poolresult2[x1i][k2i] << std::endl;
			//std::cout << pool2Filters[x1i][k2i] << std::endl;


			//std::cout << "Ǯ�� ����" << re << std::endl;
		}
		//��������Ű�� �Է�
		//4���� poolresult2�� 2���� ��� xMat���� ��ȯ
		//vec<vec<Mat>> to vec<vec<float>> ��ȯ : https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
		//vec<vec<float>> to Mat ��ȯ : https://stackoverflow.com/questions/18519647/opencv-convert-vector-of-vector-to-mat
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
	
#pragma region ��������Ű���� ����ġ ��� ������ ���
	wMat -= -(xMat.t() * (temp));
	//std::cout << wMat.size() << std::endl;
#pragma endregion
	//���Ͱ��� ���� -(yMat-yHatMat)*wMat�� pool2result ��� ũ��� ��ȯ
	std::vector<std::vector<cv::Mat>> temp2
		//= temp * wMat.t()
		;

	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			//Pooling �Լ� ������ ���
			Math::GetMaxPoolingFilter(conv2ZeroPaddingMats[x1i][k2i], pool2Filters[x1i][k2i], poolresult1[x1i][k2i], poolSize, poolStride);
			//ReLu �Լ� ������ ���
			//(conv1Mat�� ������ ��꿡�� �̹� ReLU�� ���������Ƿ� ����)
			//Math::Relu(conv1Mats[x1i][k1i], conv1Mats[x1i][k1i]);
		}
	}
#pragma region �ռ�����2 ����ġ ���(Ŀ��2) ������ ���
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
			Math::GetConvolutionKFilters(poolresult1ZeroPadding[x1i][k1i], &conv2Filters[x1i][k1i], kernels2[0][0], kernel2Stride);

			//K2 ������ ���Ϳ� ���Ͱ��� ���� ReLu�Լ� ������ �������� ����� K��� ũ��� ��ȯ
			
		}
		//kernels2[x1i][k2i] += 
		//std::cout << kernels2[x1i][k2i] << std::endl;
	}

#pragma endregion


	
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
		//BackPropagation();
	}
}
