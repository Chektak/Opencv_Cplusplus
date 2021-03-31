#include "framework.h"
int main() {
	OpencvPractice op;

	std::cout << "Hello OpenCV" << CV_VERSION << std::endl;

	//op.Camera_In();
	//op.Video_In("Resources/Video/Anime.mp4", 0.25f);
	//op.Camera_In_Video_Out("aaa.avi");
	//op.FaceScan();

	//read MNIST iamge into OpenCV Mat vector
	std::vector<cv::Mat> trainingVec;
	std::vector<uchar> labelVec;
	op.MnistTrainingDataRead("Resources/train-images.idx3-ubyte", trainingVec, USEDATA_NUM);
	op.MnistLabelDataRead("Resources/train-labels.idx1-ubyte", labelVec, USEDATA_NUM);
	op.MatPrint(trainingVec, labelVec);

	//����Ʈ�ƽ� �Լ� �׽�Ʈ
	//cv::Mat input = cv::Mat_<float>({2, 3}, {1,3,5,7,9,-600});
	//cv::Mat w = cv::Mat_<float>({3, 2}, {0, 2, 0, 2, 0, 2});
	//cv::Mat output;
	//std::cout << "�Է� ������ :\n" << input << std::endl;
	//Math::Relu(input, output);
	//std::cout << "ReLu ���� �� :\n" << output << std::endl;
	//Math::SoftMax(output, output);
	//std::cout << "����Ʈ�ƽ� ���� �� :\n" << output << std::endl;
	//Math::NeuralNetwork(output, output, w);
	//std::cout << "�Ű�� ����ġ ��� :\n" << w << std::endl;
	//std::cout << "�Ű��(��İ�) ���� �� :\n" <<output << std::endl;

	//���� ��� ����
	/*cv::Mat input = cv::Mat_<float>({ 3, 3 }, { 1,2,3,4,5,6,7,8,9 });
	cv::Mat kernel = cv::Mat_<float>({ 3, 3 }, { 0,1,0,0,1,0,0,1,0 });
	cv::Mat out;
	cv::Mat zeroPadding;
	std::cout << "�Է� ������ :\n" << input << std::endl;
	Math::CreateZeroPadding(input, zeroPadding, input.size(), kernel.size(), cv::Size(1, 1));
	std::cout << "���� �е� �� :\n" << zeroPadding << std::endl;
	Math::Convolution(zeroPadding, out, input.size(), kernel, cv::Size(1, 1));
	std::cout << "�ռ��� ���� �� :\n" << out << std::endl;
	Math::MaxPooling(out, out, cv::Size(2, 2), cv::Size(2, 2));
	std::cout << "Ǯ�� ���� �� :\n" << out << std::endl;*/

	//����Ʈ ���� �׽�Ʈ
	//std::cout << "10���� 2144444444 = 2������ "<<std::bitset<32>(2144444444) << std::endl;
	//std::cout << "10���� 2144444444�� ReverseInt 2������ " <<std::bitset<32>(op.ReverseInt(2144444444)) << std::endl;
	
	
	return 0;
}
