#include "CNNMachine.h"

void CNNMachine::Training(int epoch, double learningRate, double l2)
{
	for (int i = 0; i < epoch; i++) {
		std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
		std::cout << i<<"��° �Ʒ�" << std::endl;
		std::cout << "������ ���" << std::endl;
		ForwardPropagation();
		cost = 0;
		for (int y = 0; y < yMat.rows; y++) {
			for (int x = 0; x < yMat.cols; x++) {
				//log(0) ���� ���Ѵ� ����ó���� 0 ��� 0�� ����� �� ���
				cost += yMat.at<double>(y, x) * log((yHatMat.at<double>(y, x) == 0) ? 0.00000000001 : yHatMat.at<double>(y, x));
			}
		}
		//cost /= -yMat.rows;
		cost *= -1;
		//std::cout << i<<"yMat : " << yMat << std::endl;
		//std::cout << "yHatMat : " << yHatMat << std::endl;
		std::cout << "�ڽ�Ʈ : " << cost << std::endl;

		//�ƹ�Ű�� ������ ����
		int key = cv::waitKey(0);
		if (key != -1) {
			if (key == 13) //enterŰ
			{
				std::cout << "������ ��꿡�� ���� yMat, yHatMat, yLoss�� ������ ��� ��. " << std::endl;
				std::cout << "yMat(���� ���) : " << std::endl;
				std::cout << yMat << std::endl;
				std::cout << "yHatMat(���� ���) : " << std::endl;
				std::cout << yHatMat << std::endl;
				std::cout << "yLoss (= -(yMat - yHatMat)) : " << std::endl;
				std::cout << yLoss << std::endl;
			}
			std::cout << "������ ���" << std::endl;
			BackPropagation(learningRate);
			continue;
		}
	}
}

void CNNMachine::Init(std::vector<cv::Mat>& trainingVec, std::vector<uint8_t>& labelVec)
{
	for (int i = 0; i < trainingVec.size(); i++) {
		trainingMats.push_back(cv::Mat());
		//uchar�� ��� ��Ҹ� double�� ��� ��ҷ� Ÿ�� ĳ����
		trainingVec[i].convertTo(trainingMats[i], CV_64FC1);
	}

	poolStride = cv::Size(2, 2);
	poolSize = cv::Size(2, 2);
#pragma region ��� ����ġ ����� �յ� ������ ���� �ʱ�ȭ, Ŀ�� ������ ��� ���� �ʱ�ȭ
	cv::RNG gen(cv::getTickCount());

	//Ŀ�� 1�� ä�� �Ѱ�(�Է��� ä���� ��� ����)
	kernels1.push_back(std::vector<cv::Mat>());
	for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
		kernels1[0].push_back(cv::Mat(cv::Size(3, 3), CV_64FC1));
		gen.fill(kernels1[0][k1i], cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(1));

		kernels2.push_back(std::vector<cv::Mat>());
		//Ŀ�� 2�� ä���� Ŀ�� 1�� ����
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			kernels2[k1i].push_back(cv::Mat(cv::Size(3, 3), CV_64FC1));
			gen.fill(kernels2[k1i][k2i], cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(1));
		}
		
		
	}
	kernel1Stride = cv::Size(1, 1);
	kernel2Stride = cv::Size(1, 1);
	
	//�ռ����� ���� �е����� �����ϹǷ� Ǯ���� 2�������� ��Ҹ� ���
	int wHeight = (trainingMats[0].rows - poolSize.height) / poolStride.height + 1;
		wHeight = (wHeight - poolSize.height) / poolStride.height + 1;
	int wWidth = (trainingMats[0].cols - poolSize.width) / poolStride.width + 1;
		wWidth = (wWidth - poolSize.width) / poolStride.width + 1;

	wMat.create(cv::Size(CLASSIFICATIONNUM, wHeight * wWidth * KERNEL2_NUM), CV_64FC1);
	gen.fill(wMat, cv::RNG::UNIFORM, cv::Scalar(-1), cv::Scalar(1));

	
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
			conv1Mats[i].push_back(cv::Mat_<double>());
			conv1ZeroPaddingMats[i].push_back(cv::Mat_<double>());
			poolresult1[i].push_back(cv::Mat_<double>());
			poolresult1ZeroPadding[i].push_back(cv::Mat_<double>());
		
			pool1BackpropFilters[i].push_back(cv::Mat_<double>());
		}
		for (int j = 0; j < KERNEL2_NUM; j++) {
			conv2Mats[i].push_back(cv::Mat_<double>());
			conv2ZeroPaddingMats[i].push_back(cv::Mat_<double>());
			poolresult2[i].push_back(cv::Mat_<double>());
			
			pool2BackpropFilters[i].push_back(cv::Mat_<double>());
		}
	}
#pragma endregion

#pragma region �ռ��� ������ ��� ���� �ʱ�ȭ
	//�ռ��� �� ���� �е��� ����ϹǷ� Ǯ�� ��� ũ�⸸ ���
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
	//1��° �ռ����� ������ ��� ���� �ʱ�ȭ
	int r1Size = trainingMats[0].rows * trainingMats[0].cols;
	for (int r1i = 0; r1i < r1Size; r1i++) {
		conv1BackpropFilters.push_back(std::pair<int, int>());
	}
	//2��° �ռ��� Ŀ���� ������ ��� ���� �ʱ�ȭ
	int r2Size = pool1ResultSize.width * pool1ResultSize.height;
	for (int r2i = 0; r2i < r2Size; r2i++) {
		conv2BackpropFilters.push_back(std::pair<int, int>());
	}
#pragma endregion

	//���� �����͸� ���ͷ� ��ȯ�Ѵ�.
	yMat = cv::Mat::zeros(cv::Size(CLASSIFICATIONNUM, trainingMats.size()), CV_64FC1);
	for (int y = 0; y < labelVec.size(); y++) {
		//���� �´´ٸ� true(1), �ƴ϶�� false(0)�� ����
		yMat.at<double>(y, labelVec[y]) = 1;
	}

	yHatMat.create(cv::Size(CLASSIFICATIONNUM, trainingMats.size()), CV_64FC1);


	//lossAverage = 2305843009213693951;
}

void CNNMachine::ForwardPropagation()
{
	//�ռ����� ��� ����� ��������Ű�� �Է����� ��ȯ�� �� ���
	std::vector<std::vector<double>> tempArr;
	cv::Mat tempMat;

	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		tempArr.push_back(std::vector<double>());
		x1ZeroPaddingMats.push_back(cv::Mat_<double>());

		Math::CreateZeroPadding(trainingMats[x1i], x1ZeroPaddingMats[x1i], trainingMats[0].size(), kernels1[0][0].size(), kernel1Stride);
		//�ռ����� 1
		for (int k1i = 0; k1i < KERNEL1_NUM; k1i++) {
			Math::Convolution(x1ZeroPaddingMats[x1i], conv1Mats[x1i][k1i], trainingMats[0].size(), kernels1[0][k1i], kernel1Stride);
			Math::Relu(conv1Mats[x1i][k1i], conv1Mats[x1i][k1i]);
			Math::CreateZeroPadding(conv1Mats[x1i][k1i], conv1ZeroPaddingMats[x1i][k1i], pool1ResultSize, poolSize, poolStride);
			Math::MaxPooling(conv1ZeroPaddingMats[x1i][k1i], poolresult1[x1i][k1i], poolSize, poolStride);

			Math::CreateZeroPadding(poolresult1[x1i][k1i], poolresult1ZeroPadding[x1i][k1i], poolresult1[0][0].size(), kernels2[0][0].size(), kernel2Stride);
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
		}
		//��������Ű�� �Է�
		//4���� poolresult2�� 2���� ��� xMat���� ��ȯ
		//vec<vec<Mat>> to vec<vec<double>> ��ȯ : https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
		//vec<vec<double>> to Mat ��ȯ : https://stackoverflow.com/questions/18519647/opencv-convert-vector-of-vector-to-mat
		for (int k2i = 0; k2i < KERNEL2_NUM; k2i++) {
			tempMat = poolresult2[x1i][k2i];
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

	Math::NeuralNetwork(xMat, yHatMat, wMat);
	Math::SoftMax(yHatMat, yHatMat);
}

void CNNMachine::BackPropagation(double learningRate)
{
	yLoss = -(yMat - yHatMat); //�ս��Լ��� SoftMax �Լ� ����� ���� �̺��� ��
	wT = wMat.t();
	yLossW = yLoss*wT; //�ս� �Լ��� ���������� W�� ���� �̺��� ��
	std::vector<std::vector<cv::Mat>> yLossWTemp; //yLossW�� Ǯ��2�������� ũ��� ���� ��ȯ

	std::vector<std::vector<cv::Mat>> yLossWUpRelu2; //�ս� �Լ��� �ռ���2 ����� ���� �̺��� �� (Up�� Up-Sampleling(Ǯ���Լ��� �̺�) ����)
	std::vector<std::vector<cv::Mat>> yLossWUpRelu2P1UpRelu; //�ս� �Լ��� �ռ���1 ����� ���� �̺��� ��


	//���Ͱ��� ���� yLossW�� Ǯ��2 ��� ��� ũ��� ��ȯ
	for (int i = 0; i < trainingMats.size(); i++) {
		yLossWTemp.push_back(std::vector<cv::Mat>());
		for (int j = 0; j < KERNEL2_NUM; j++) {
			cv::Mat sample = yLossW.row(i).reshape(1, poolresult2[0].size()).row(j).reshape(1, poolresult2[0][0].rows);
			yLossWTemp[i].push_back(sample);
		}

	}
	
	//std::cout << wT << std::endl;

	//���� ��ȯ�� yLossW�� Ǯ��2 ���ͷ� Up-Sampleling �� Relu(Conv2)��İ� ���Ͱ�
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		yLossWUpRelu2.push_back(std::vector<cv::Mat>());
		for (int k2n = 0; k2n < KERNEL2_NUM; k2n++) {
			yLossWUpRelu2[x1i].push_back(cv::Mat());
			//Pooling �Լ� ������ ������� Ǯ�� ���� �Ҵ�
			Math::GetMaxPoolingFilter(conv2ZeroPaddingMats[x1i][k2n], pool2BackpropFilters[x1i][k2n], poolresult2[x1i][k2n], poolSize, poolStride);
			//Ǯ�� ���ͷ� �����ø�
			Math::MaxPoolingBackprop(yLossWTemp[x1i][k2n], yLossWUpRelu2[x1i][k2n], pool2BackpropFilters[x1i][k2n], poolSize, poolStride);

			//Relu �Լ� ������ ���
			//(������ ��꿡�� �̹� ReLU�� ���������Ƿ� ����)
			//Math::Relu(conv2Mats[x1i][k2n], conv2Mats[x1i][k2n]);

			//Up-Sampleling ��� ��İ� Relu(Conv2)����� ���Ͱ�
			yLossWUpRelu2[x1i][k2n] = yLossWUpRelu2[x1i][k2n].mul(conv2Mats[x1i][k2n]);
			//Math::Relu(yLossWUpRelu2[x1i][k2n], yLossWUpRelu2[x1i][k2n]);
			//std::cout << "Conv2Mats\n" << conv2Mats[x1i][k2n] << std::endl;
			//std::cout << "ReLu Conv2���� ���Ͱ� ��\n" << yLossWUpRelu2[x1i][k2n] << std::endl;
		}
	}
	
	//Ŀ��2 ������ ����� ���� �ռ���2 ���� ���
	Math::GetConvBackpropFilters(poolresult1[0][0], &conv2BackpropFilters, kernels2[0][0], kernel2Stride);
	//Ŀ��1 ������ ����� ���� �ռ���1 ���� ���
	Math::GetConvBackpropFilters(trainingMats[0], &conv1BackpropFilters, kernels1[0][0], kernel1Stride);

	//yLossWUpRelu2��İ� �ռ���2 �Լ��� Ŀ��2�� ���� �̺� ����� ���Ͱ��ϰ�, Ǯ��1 ���ͷ� Up-Sampleling �� Relu(Conv1)��İ� ���Ͱ�
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		yLossWUpRelu2P1UpRelu.push_back(std::vector<cv::Mat>());
		//Ŀ�� 1 ������ŭ �ݺ�
		for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
			yLossWUpRelu2P1UpRelu[x1i].push_back(cv::Mat());
			cv::Mat yLossWUpRelu2P1;

			//Ŀ�� 2 ������ŭ �ݺ�
			for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
				//yLossWUpRelu2��İ� �ռ���2 �Լ��� Ŀ��2�� ���� �̺� ����� ���Ͱ�
				Math::ConvXBackprop(yLossWUpRelu2[x1i][k2n], kernels2[k1n][k2n], yLossWUpRelu2P1, conv2BackpropFilters, kernel1Stride, learningRate);
			}
			//Pooling �Լ� ������ ������� Ǯ�� ���� ����
			Math::GetMaxPoolingFilter(conv1ZeroPaddingMats[x1i][k1n], pool1BackpropFilters[x1i][k1n], poolresult1[x1i][k1n], poolSize, poolStride);
			//Ǯ�� ���ͷ� �����ø�
			Math::MaxPoolingBackprop(yLossWUpRelu2P1, yLossWUpRelu2P1UpRelu[x1i][k1n], pool1BackpropFilters[x1i][k1n], poolSize, poolStride);

			//Relu �Լ� ������ ���
			//(������ ��꿡�� �̹� ReLU�� ���������Ƿ� ����)
			//Math::Relu(conv1Mats[x1i][k1n], conv1Mats[x1i][k1n]);
			yLossWUpRelu2P1UpRelu[x1i][k1n] = yLossWUpRelu2P1UpRelu[x1i][k1n].mul(conv1Mats[x1i][k1n]);
		}
	}
#pragma region �ռ�����1 ����ġ ���(Ŀ��1) ������ ���
	//std::cout << "\nĿ�� 1 ������ ��� " << std::endl;
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		for (int k1c = 0; k1c < kernels1.size(); k1c++) {
			for (int k1n = 0; k1n < kernels1[0].size(); k1n++) {
				cv::Mat newKernel;
				Math::ConvKBackprop(-yLossWUpRelu2P1UpRelu[x1i][k1n], x1ZeroPaddingMats[x1i], kernels1[k1c][k1n], newKernel, conv1BackpropFilters, kernel1Stride, learningRate);
				newKernel.copyTo(kernels1[k1c][k1n]);
				//std::cout << "Ŀ�� ������ ��� ������Ʈ �� : " << kernels1[k1c][k1n] << std::endl;
			}
		}
	}
#pragma endregion

#pragma region �ռ�����2 ����ġ ���(Ŀ��2) ������ ���
	//std::cout << "\nĿ�� 2 ������ ��� " << std::endl;
	for (int x1i = 0; x1i < trainingMats.size(); x1i++) {
		for (int k2c = 0; k2c < kernels2.size(); k2c++) {
			for (int k2n = 0; k2n < kernels2[0].size(); k2n++) {
				cv::Mat newKernel;

				Math::ConvKBackprop(-yLossWUpRelu2[x1i][k2n], poolresult1ZeroPadding[x1i][k2c], kernels2[k2c][k2n],newKernel, conv2BackpropFilters, kernel2Stride, learningRate);
				newKernel.copyTo(kernels2[k2c][k2n]);
				//std::cout << "Ŀ�� ������ ��� ������Ʈ �� : " << kernels2[k2c][k2n] << std::endl;
			}
		}
	}

#pragma endregion

#pragma region ��������Ű���� ����ġ ��� ������ ���
	wMat -= learningRate *(xMat.t() * (yLoss));
	//std::cout << wMat << std::endl;
#pragma endregion
	
	
}


