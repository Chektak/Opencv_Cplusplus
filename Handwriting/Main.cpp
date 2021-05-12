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
	
	//std::cout << "����� �Ʒ� ������ ���� �Է����ּ���." << std::endl;
	//std::cin >> USEDATA_NUM;
	//std::cout << "����� Ŀ��1 ���� �Է����ּ���." << std::endl;
	//std::cin >> KERNEL1_NUM;
	//std::cout << "����� Ŀ��2 ���� �Է����ּ���." << std::endl;
	//std::cin >> KERNEL2_NUM;
	//op.PaintWindow(cv::Mat(), cv::Mat(), "Paint", cv::Size(560, 560), 13, NULL);

	std::vector<cv::Mat> imageMats;
	std::vector<uint8_t> labelVec;
	//op.MnistImageMatDataRead("Resources/train-images.idx3-ubyte", imageMats, 0, USEDATA_NUM);
	//op.MnistImageLabelDataRead("Resources/train-labels.idx1-ubyte", labelVec, 0, USEDATA_NUM);
	//op.MatPrint(imageMats, labelVec);
	CNNMachine cnn;
	cnn.Init(&op, USEDATA_NUM, KERNEL1_NUM, KERNEL2_NUM, CLASSIFICATIONNUM);
	//�Ҽ��� 15�ڸ����� ���
	std::cout << std::fixed;
	std::cout.precision(15);
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
