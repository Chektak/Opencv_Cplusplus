#include "CNNMachine.h"

void CNNMachine::Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec)
{
	for (int i = 0; i < trainingVec.size(); i++) {
		trainingMats.push_back(cv::Mat());
		//uchar�� ��� ��Ҹ� float�� ��� ��ҷ� Ÿ�� ĳ����
		trainingVec[i].convertTo(trainingMats[i], CV_32FC1);
	}

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

#pragma region �ռ��� ������ ��� ���� �ʱ�ȭ
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

	//1��° �ռ����� ������ ��� ���� �ʱ�ȭ
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		conv1BackpropFilters.push_back(std::vector<std::vector<std::pair<int, int>>>());
		//Ŀ�� 1�� ä�� �Ѱ�(�Է��� ä���� ��� ����)
		for (int k1c = 0; k1c < kernels1.size(); k1c++) {
			conv1BackpropFilters[x1i].push_back(std::vector<std::pair<int, int>>());
			//�ռ��� 1�� ��� ��� ũ�⸸ŭ �ʱ�ȭ
			int r1Size = trainingMats[0].rows * trainingMats[0].cols;
			for (int r1i = 0; r1i < r1Size; r1i++) {
				conv1BackpropFilters[x1i][k1c].push_back(std::pair<int, int>());
			}
		}
	}
	//2��° �ռ��� Ŀ���� ������ ��� ���� �ʱ�ȭ
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		conv2BackpropFilters.push_back(std::vector<std::vector<std::pair<int, int>>>());
		//Ŀ�� 2�� ä�� ������ Ŀ��1�� ����
		for (int k2c = 0; k2c < kernels1[0].size(); k2c++) {
			conv2BackpropFilters[x1i].push_back(std::vector<std::pair<int, int>>());
			//�ռ��� 2�� ��� ��� ũ�⸸ŭ �ʱ�ȭ 
			int r2Size = pool1ResultSize.width * pool1ResultSize.height;
			for (int r2i = 0; r2i < r2Size; r2i++) {
				conv2BackpropFilters[x1i][k2c].push_back(std::pair<int, int>());
			}
		}
	}
#pragma endregion

	//���� �����͸� ���ͷ� ��ȯ�Ѵ�.
	yMat = cv::Mat::zeros(cv::Size(CLASSIFICATIONNUM, trainingMats.size()), CV_32FC1);
	for (int y = 0; y < labelVec.size(); y++) {
		//���� �´´ٸ� true(1), �ƴ϶�� false(0)�� ����
		yMat.at<float>(y, labelVec[y]) = 1;
	}

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
		tempArr.push_back(std::vector<float>());
		x1ZeroPaddingMats.push_back(cv::Mat_<float>());

		Math::CreateZeroPadding(trainingMats[x1i], x1ZeroPaddingMats[x1i], trainingMatrixSize, k1MatrixSize, kernel1Stride);

		//�ռ����� 1
		for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
			Math::Convolution(x1ZeroPaddingMats[x1i], conv1Mats[x1i][k1i], trainingMatrixSize, kernels1[0][k1i], kernel1Stride);
			Math::Relu(conv1Mats[x1i][k1i], conv1Mats[x1i][k1i]);
			Math::CreateZeroPadding(conv1Mats[x1i][k1i], conv1ZeroPaddingMats[x1i][k1i], pool1ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv1ZeroPaddingMats[x1i][k1i], poolresult1[x1i][k1i], poolSize, poolStride);

			Math::CreateZeroPadding(poolresult1[x1i][k1i], poolresult1ZeroPadding[x1i][k1i], poolresult1[0][0].size(), k2MatrixSize, kernel2Stride);
		}
		//�ռ����� 2
		/*�ռ����� 1�� (��:������ ��, ��:ä�� ��)�� �̹����� ���� poolresult1��İ�
		�ռ����� 2�� kernel2����� ��İ��ϵ� ����*/
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
				Math::Convolution(poolresult1ZeroPadding[x1i][k1i], conv2Mats[x1i][k2i], poolresult1[0][0].size(), kernels2[k1i][k2i], kernel2Stride);
			}
			Math::Relu(conv2Mats[x1i][k2i], conv2Mats[x1i][k2i]);
			Math::CreateZeroPadding(conv2Mats[x1i][k2i], conv2ZeroPaddingMats[x1i][k2i], pool2ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv2ZeroPaddingMats[x1i][k2i], poolresult2[x1i][k2i], poolSize, poolStride);
			//Math::GetMaxPoolingFilter(conv2Mats[x1i][k2i], pool2BackpropFilters[x1i][k2i], poolresult2[x1i][k2i], poolSize, poolStride);
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

	Math::NeuralNetwork(xMat, yHatMat, wMat);
	Math::SoftMax(yHatMat, yHatMat);
}

void CNNMachine::BackPropagation()
{
	cv::Mat yLoss = -(yMat - yHatMat);
	cv::Mat wT = wMat.t();
	cv::Mat yLossW = yLoss*wT;
	std::cout << "yLossW size = " <<yLossW.size()<<std::endl;
	//���Ͱ��� ���� yLossW�� pool2result ��� ũ��� ��ȯ
	std::vector<std::vector<cv::Mat>> yLossWMats;
	//�Ʒ� ������ ��(= USEDATA_NUM)��ŭ �ݺ�
	for (int i = 0; i < poolresult2.size(); i++) {
		yLossWMats.push_back(std::vector<cv::Mat>());
		//poolresult2�� ä�� ��(= KERNEL2_NUM)��ŭ �ݺ�
		for (int j = 0; j < poolresult2[0].size(); j++) {
			cv::Mat sample = yLossW.row(i).reshape(1, poolresult2[0].size()).row(j).reshape(1, poolresult2[0][0].rows);
			yLossWMats[i].push_back(sample);
		}
	}
	std::vector<std::vector<cv::Mat>> yLossWPoolUpMats;

	//��ȯ�� yLossW�� Ǯ�� ���ͷ� Up-Sampleling �� ReLu(Conv)��İ� ���Ͱ�
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		yLossWPoolUpMats.push_back(std::vector<cv::Mat>());
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			yLossWPoolUpMats[x1i].push_back(cv::Mat());

			//Pooling �Լ� ������ ���
			Math::GetMaxPoolingFilter(conv2ZeroPaddingMats[x1i][k2i], pool2BackpropFilters[x1i][k2i], poolresult2[x1i][k2i], poolSize, poolStride);
			
			//std::cout << pool2BackpropFilters[x1i][k2i].size() << std::endl;
			//std::cout << pool2BackpropFilters[x1i][k2i] << std::endl;
			//std::cout << yLossWMats[x1i][k2i].size() << std::endl;
			//std::cout << yLossWMats[x1i][k2i] << std::endl;

			//Ǯ�� ���ͷ� �����ø�
			Math::MaxPoolingBackprop(yLossWMats[x1i][k2i], yLossWPoolUpMats[x1i][k2i], pool2BackpropFilters[x1i][k2i], poolSize, poolStride);

			//std::cout << yLossWPoolUpMats[x1i][k2i] << std::endl;
			//std::cout << conv2Mats[x1i][k2i] << std::endl;
			
			//ReLu �Լ� ������ ���
			//(������ ��꿡�� �̹� ReLU�� ���������Ƿ� ����)
			//Math::Relu(conv2Mats[x1i][k2i], conv2Mats[x1i][k2i]);
			
			//Up-Sampleling ��� ��İ� ReLu(Conv)����� ���Ͱ�
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

#pragma region �ռ�����1 ����ġ ���(Ŀ��1) ������ ���
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		//Ŀ�� 1 ä�� ����ŭ �ݺ�
		for (int k1c = 0; k1c < kernels1.size(); k1c++) {
			//�ռ����� Ŀ�ο� ���� �̺� (Ŀ�� ���� ���)
			std::cout << "�ռ���1 �Է� ��� ũ�� : " << trainingMats[x1i].size() << ", �ռ���1 �����е� �Է� ��� ũ�� : " << x1ZeroPaddingMats[x1i].size() <<  ", ���� ��� ũ�� : " << conv1BackpropFilters[x1i][k1c].size() << std::endl;
			
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

#pragma region �ռ�����2 ����ġ ���(Ŀ��2) ������ ���
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		//Ŀ�� 2 ä�� ����ŭ �ݺ�
		for (int k2c = 0; k2c < kernels2.size(); k2c++) {
			//�ռ����� Ŀ�ο� ���� �̺� (Ŀ�� ���� ���)
			std::cout << "�ռ���2 �Է� ��� ũ�� : " << poolresult1[x1i][k2c].size() << "," << "���� ��� ũ�� : " << conv2BackpropFilters[x1i][k2c].size() << std::endl;
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

#pragma region ��������Ű���� ����ġ ��� ������ ���
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
