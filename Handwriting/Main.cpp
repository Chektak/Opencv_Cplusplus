#include "framework.h"
int main() {
	OpencvPractice op;

	std::cout << "Hello OpenCV" << CV_VERSION << std::endl;

	//op.Camera_In();
	//op.Video_In("Resources/Video/Anime.mp4", 0.25f);
	//op.Camera_In_Video_Out("aaa.avi");
	//op.FaceScan();

	//����Ʈ�ƽ� �Լ� �׽�Ʈ
	/*cv::Mat input = cv::Mat_<double>({2, 3}, {1,3,5,7,9,-600});
	cv::Mat w = cv::Mat_<double>({3, 2}, {0, 2, 0, 2, 0, 2});
	cv::Mat output;
	std::cout << "�Է� ������ :\n" << input << std::endl;
	Math::Relu(input, output);
	std::cout << "ReLu ���� �� :\n" << output << std::endl;
	Math::SoftMax(output, output);
	std::cout << "����Ʈ�ƽ� ���� �� :\n" << output << std::endl;
	Math::NeuralNetwork(output, output, w);
	std::cout << "�Ű�� ����ġ ��� :\n" << w << std::endl;
	std::cout << "�Ű��(��İ�) ���� �� :\n" <<output << std::endl;*/

	//����Ʈ ���� �׽�Ʈ
	//std::cout << "10���� 2144444444 = 2������ "<<std::bitset<32>(2144444444) << std::endl;
	//std::cout << "10���� 2144444444�� ReverseInt 2������ " <<std::bitset<32>(op.ReverseInt(2144444444)) << std::endl;
	
	//���� ��� ����
	/*cv::Mat input = cv::Mat_<double>({ 3, 3 }, { 1,2,3,4,5,6,7,8,9 });
	cv::Mat kernel = cv::Mat_<double>({ 3, 3 }, { 0,1,0,0,1,0,0,1,0 });
	cv::Mat out;
	cv::Mat zeroPadding;
	std::cout << out.type() << std::endl;
	std::cout << zeroPadding.type() << std::endl;
	std::cout << input.type() << std::endl;
	std::cout << kernel.type() << std::endl;
	std::cout << "�Է� ������ :\n" << input << std::endl;
	Math::CreateZeroPadding(input, zeroPadding, input.size(), kernel.size(), cv::Size(1, 1));
	std::cout << "���� �е� �� :\n" << zeroPadding << std::endl;
	Math::Convolution(zeroPadding, out, input.size(), kernel, cv::Size(1, 1));
	std::cout << "�ռ��� ���� �� :\n" << out << std::endl;
	Math::MaxPooling(out, out, cv::Size(2, 2), cv::Size(2, 2));
	std::cout << "Ǯ�� ���� �� :\n" << out << std::endl;*/
	
	//read MNIST iamge into OpenCV Mat vector
	
	
	//op.PaintWindow(cv::Mat(), cv::Mat(), "Paint", cv::Size(560, 560), 13, NULL);

	CNNMachine cnn;
	bool loadModelYesOrNo;
	int useDataNum = 0;
	int kernel1Num = 0;
	int kernel2Num = 0;
	int neuralW1Cols_Num = 0;
	bool loadSucceed = false;

	std::cout << "����� ���� �ҷ����ðڽ��ϱ�? (Yes : 1, No : 0)" << std::endl;
	std::cin >> loadModelYesOrNo;
	if (loadModelYesOrNo)
	{
		loadSucceed = cnn.LoadModel("Model.json");
		if(!loadSucceed)
			std::cout << "����� �� �ҷ����⿡ �����߽��ϴ�." << std::endl;
	}
	if (!loadModelYesOrNo || !loadSucceed) {
		std::cout << "����� ��ü �����ͼ� ���� �Է����ּ���." << std::endl;
		std::cin >> useDataNum;
		std::cout << "����� �ռ��� 1�� ���� ��(Ŀ��1 ��)�� �Է����ּ���." << std::endl;
		std::cin >> kernel1Num;
		std::cout << "����� �ռ��� 2�� ���� ��(Ŀ��2 ��)�� �Է����ּ���." << std::endl;
		std::cin >> kernel2Num;
		std::cout << "����� ��������Ű�� 1�� ���� ���� �Է����ּ���." << std::endl;
		std::cin >> neuralW1Cols_Num;

		cnn.Init(&op, useDataNum, kernel1Num, kernel2Num, neuralW1Cols_Num, CLASSIFICATIONNUM);
	}
	//�Ҽ��� 15�ڸ����� ���
	std::cout << std::fixed;
	std::cout.precision(6);
	cnn.Training(-1, 0.0001, 1, CNNMachine::GD::BATCH);
	std::cout << "END" << std::endl;

	//���� ��� �Լ� ������ ���� ��� �׽�Ʈ
	//3*3 ũ���� �Է� ���, 3*3 ũ���� Ŀ�� ���, stride�� 1*1 ���
	/*cv::Mat input(cv::Size(3, 3), CV_64FC1);
	cv::Mat k(cv::Size(3, 3), CV_64FC1);
	std::vector<std::pair<int, int>> vec;
	for (int i = 0; i < 9; i++) {
		vec.push_back(std::pair<int, int>());
	}
	Math::GetConvBackpropFilters(input, &vec, k, cv::Size(1, 1));*/

	//������ �׸��� ����
	//EnterŰ�� �ƽ�Ű�ڵ�� 13
	/*cv::Mat screen = cv::Mat::zeros(cv::Size(28, 28), CV_64FC1);
	op.PaintWindow(screen, "Paint", screen.size()*20, 13);*/
	return 0;
}
