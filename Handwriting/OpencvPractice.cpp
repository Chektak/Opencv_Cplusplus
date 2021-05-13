#include "OpencvPractice.h"

void OpencvPractice::ImgSizePrint(cv::String imgPath)
{
	cv::Mat img1 = cv::imread(imgPath);
	std::cout << "�̹��� ���� : "<<img1.rows << std::endl;
	std::cout << "�̹��� ���� : "<<img1.cols << std::endl;
}
void OpencvPractice::ImgInfoPrint(cv::String imgPath)
{
	cv::Mat img1 = cv::imread(imgPath);

	std::cout << "" << img1.cols << std::endl;
	std::cout << "" << img1.rows << std::endl;
	std::cout << "" << img1.channels() << std::endl;

	if (img1.type() == CV_8UC1) {
		std::cout << "" << std::endl;
	}
	else if (img1.type() == CV_8UC3) {
		std::cout << "" << std::endl;
	}

	double data[] = { 2.f, 1.414f, 3.f, 1.732f };
	cv::Mat mat1(2, 2, CV_64FC1, data);
	std::cout << "mat1:\n" << mat1 << std::endl;
}
void OpencvPractice::RandomImage(cv::String imgPath = "Resource/Image/lenna.bmp")
{
	cv::Mat img;
	img = cv::imread(imgPath);

	if (img.empty()) {
		std::cerr << "Image load failed!" << std::endl;
		return;
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> posR(0, 1080);
	std::uniform_int_distribution<int> sizeR(100, 500);


	cv::namedWindow("Image");
	cv::imshow("Image", img);

	while (1)
	{
		cv::moveWindow("Image", posR(gen), posR(gen));
		cv::resizeWindow("Image", sizeR(gen), sizeR(gen));
		if (cv::waitKeyEx(delay) == 27)
			break;
	}
	cv::destroyAllWindows();
}

void OpencvPractice::MatOp1()
{
	cv::Mat img1;

	cv::Mat img2(480, 640, CV_8UC1);
	cv::Mat img3(480, 640, CV_8UC3);
	cv::Mat img4(cv::Size(640, 480), CV_8UC3);

	cv::Mat img5(480, 640, CV_8UC1, cv::Scalar(128));
	cv::Mat img6(480, 640, CV_8UC3, cv::Scalar(0, 0, 255));

	cv::Mat mat1 = cv::Mat::zeros(3, 3, CV_32SC1);
	cv::Mat mat2 = cv::Mat::ones(3, 3, CV_32SC1);
	cv::Mat mat3 = cv::Mat::eye(3, 3, CV_32SC1);

	double data[] = { 1, 2,3,4,5,6 };
	cv::Mat mat4(2, 3, CV_64FC1, data);

	cv::Mat mat5 = (cv::Mat_<double>(2, 3) << 1, 2, 3, 4, 5, 6);
	cv::Mat mat6 = cv::Mat_<uint8_t>({ 2,3 }, { 1,2,3,4,5,6 });

	mat4.create(256, 256, CV_8UC3);
	mat5.create(4, 4, CV_64FC1);

	mat4 = cv::Scalar(255, 0, 0);
	mat5.setTo(1.f);

}

void OpencvPractice::ImagePrint(cv::String imgName)
{
	cv::Mat img1 = cv::imread(imgName);
	if (img1.empty()) {
		//std::cerr : ���� ��¿� �ַ� ���Ǵ� ��Ʈ��
		std::cerr << "Image load failed!" << std::endl;
		return;
	}
	cv::Mat img2 = img1(cv::Rect(220, 80, 340, 280));//�� ����
	cv::Mat img3 = img1(cv::Rect(220, 80, 340, 280)).clone();//�� ����

	img2 = ~img2; //�̹��� �÷� ����

	cv::imshow("img1", img1);
	cv::imshow("img2", img2);
	cv::imshow("img3", img3);

	cv::waitKeyEx();
	cv::destroyAllWindows();
}

void OpencvPractice::Camera_In()
{
	cv::VideoCapture cap(0); //0��° ī�޶� ����
	if (!cap.isOpened())
	{
		std::cerr << "Camera open failed!" << std::endl;
		return;
	}
	std::cout << "Frame width: " << cvRound(cap.get(cv::CAP_PROP_FRAME_WIDTH)) << std::endl;
	std::cout << "Frame height: " << cvRound(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) << std::endl;

	cv::Mat frame, inversed;
	while (true) {
		if (!cap.read(frame))
			break;
		
		inversed = ~frame;

		cv::imshow("basic", frame);
		cv::imshow("inversed", inversed);
		

		if (cv::waitKeyEx(delay) == 27) //ESC Ű
			break;
	}
	cv::destroyAllWindows();
}

void OpencvPractice::Video_In(cv::String videoPath, double winScale)
{
	cv::VideoCapture cap(videoPath);
	
	if (!cap.isOpened()) {
		std::cerr << "Video open failed!" << std::endl;
		return;
	}

	std::cout << "Frame width: " << cvRound(cap.get(cv::CAP_PROP_FRAME_WIDTH)) << std::endl;
	std::cout << "Frame height: " << cvRound(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) << std::endl;
	std::cout << "Frame count: " << cvRound(cap.get(cv::CAP_PROP_FRAME_COUNT)) << std::endl;

	cv::namedWindow("Basic", cv::WINDOW_NORMAL);
	cv::namedWindow("Inversed", cv::WINDOW_NORMAL);
	cv::resizeWindow("Basic", (int)(cap.get(cv::CAP_PROP_FRAME_WIDTH) * winScale), (int)(cap.get(cv::CAP_PROP_FRAME_HEIGHT) * winScale));
	cv::resizeWindow("Inversed", (int)(cap.get(cv::CAP_PROP_FRAME_WIDTH) * winScale),(int)(cap.get(cv::CAP_PROP_FRAME_HEIGHT) * winScale));

	cv::Mat frame, inversed;
	while (true) {
		if (!cap.read(frame))
			break;
		
		inversed = ~frame;

		cv::imshow("Basic", frame);
		cv::imshow("Inversed", inversed);
		if (cv::waitKeyEx(delay) == 27)
			break;
	}
	cv::destroyAllWindows();
	
	//VideoCaptureŬ������ �Ҹ� �� �� �Ҹ��ڿ��� �޸� �ڵ� ����
}

void OpencvPractice::Camera_In_Video_Out(cv::String savePath)
{
	cv::VideoCapture cap(0);

	if (!cap.isOpened()) {
		std::cerr << "Camera open failed!" << std::endl;
		return;
	}

	int w = cvRound(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int h = cvRound(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

	int fourcc = cv::VideoWriter::fourcc('F', 'M', 'P', '4');
	cv::VideoWriter outputVideo(savePath, fourcc, fps, cv::Size(w, h));

	cv::Mat frame, inversed;

	while (true) {
		if (!cap.read(frame)) {
			break;
		}
		inversed = ~frame;
		outputVideo.write(inversed);
		
		cv::imshow("frame", frame);
		cv::imshow("inversed", inversed);
		
		if (cv::waitKeyEx(delay) == 27)
			break;
	}
	cv::destroyAllWindows();
}

void OpencvPractice::FaceScan()
{
	cv::VideoCapture cap(0); //0��° ī�޶� ����
	if (!cap.isOpened())
	{
		std::cerr << "Camera open failed!" << std::endl;
		return;
	}

	cv::Mat frame;
	cv::CascadeClassifier faceClassifier;
	faceClassifier.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");

	cv::namedWindow("webcam", 1);

	while (true) {
		if (!cap.read(frame))
			break;
		cv::Mat grayscale;
		cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(grayscale, grayscale);
		std::vector<cv::Rect> faces;
		faceClassifier.detectMultiScale(grayscale, faces, 1.1, 3,
			0,
			//cv::CASCADE_FIND_BIGGEST_OBJECT | cv::CASCADE_SCALE_IMAGE, 
			cv::Size(30, 30));

		for (int i = 0; i < faces.size(); i++) {
			cv::Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			cv::Point tr(faces[i].x, faces[i].y);

			cv::rectangle(frame, lb, tr, cv::Scalar(0, 255, 0), 3, 4, 0);

			cv::imshow("webcam", frame);
		}

		if (cv::waitKeyEx(200) == 27) //ESC Ű
			break;
	}
	cv::destroyAllWindows();
}

void OpencvPractice::MatPrint(std::vector<cv::Mat>& imageMats, std::vector<uint8_t>& labelVec)
{
	std::cout << "�о�� �Ʒ� ������ �� : " << imageMats.size() << std::endl;
	std::cout << "�о�� ���� ������ �� : " << labelVec.size() << std::endl;

	cv::namedWindow("Window", cv::WINDOW_NORMAL);

	for (int i = 0; i < labelVec.size() && i < imageMats.size(); i++) {
		imshow("HandWriting", imageMats[i]);
		//std::cout << i << "��° �̹��� ���� : " << (int)labelVec[i] <<std::endl;
		//�ƹ� Ű�� ������ ����
		/*if (cv::waitKeyEx(0) != -1)
			continue;*/
	}
}

//32bit �������� �б� ���� ��Ʋ ����� CPU���� ���ġ�ϴ� �Լ�
//1����Ʈ(8��Ʈ)�� �Ųٷ� �迭�� int�� ��ȯ
int OpencvPractice::ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void OpencvPractice::PaintWindow(cv::InputOutputArray paintMat, cv::String windowName, cv::Size windowSize, int exitAsciiCode)
{
	cv::Mat inputScreen = paintMat.getMat();
	if(inputScreen.empty())
		inputScreen = cv::Mat::zeros(cv::Size(28, 28), CV_64FC1);
	inputScreen.setTo(0);

	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::imshow(windowName, inputScreen);
	cv::setMouseCallback(windowName, CallBackFunc, &inputScreen);
	cv::resizeWindow(windowName, windowSize);
	while (cv::waitKeyEx(1) != exitAsciiCode) {
		cv::imshow(windowName, inputScreen);
	}
}

//static ���� �ʱ�ȭ (http://egloos.zum.com/kaludin/v/2461942)
cv::Point OpencvPractice::mousePt(0, 0);
bool OpencvPractice::mouseLeftPress = false;
bool OpencvPractice::mouseRightPress = false;

void OpencvPractice::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	std::cout << "pos (" << x << "," << y << ") ";
	mousePt.x = x;
	mousePt.y = y;
	
	//���� �͸��Լ� ���
	auto drawCircle = [](int x, int y, void* userdata) {

		cv::Mat& img = *((cv::Mat*)(userdata)); // 1st cast it back, then deref
		//opencv ���� �׸��� �Լ� ���� : https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html
		//���� ���� : https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html
		//cv::circle(img, cv::Point(x, y), 2, cv::Scalar(0.8), cv::FILLED, cv::LineTypes::LINE_4);
		cv::circle(img, cv::Point(x, y), 1, cv::Scalar(1), cv::FILLED, cv::LineTypes::LINE_4);
	};
	
	auto eraserCircle = [](int x, int y, void* userdata) {

		cv::Mat& img = *((cv::Mat*)(userdata)); // 1st cast it back, then deref
		//opencv ���� �׸��� �Լ� ���� : https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html
		//���� ���� : https://docs.opencv.org/4.5.1/d6/d6e/group__imgproc__draw.html
		cv::circle(img, cv::Point(x, y), 1, cv::Scalar(0), cv::FILLED, cv::LineTypes::LINE_4);
	};
	
	switch (event)
	{
	case cv::EVENT_MOUSEMOVE:
	{
		std::cout << "cv::EVENT_MOUSEMOVE" << std::endl;

		if (mouseLeftPress == true)
		{
			drawCircle(x, y, userdata);
		}
		if (mouseRightPress == true) {
			eraserCircle(x, y, userdata);
		}
		break;
	}
	case cv::EVENT_LBUTTONDOWN:
	{
		std::cout << "cv::EVENT_LBUTTONDOWN" << std::endl;

		mouseLeftPress = true;
		drawCircle(x, y, userdata);
		break;
	}
	case cv::EVENT_LBUTTONUP:
	{
		std::cout << "cv::EVENT_LBUTTONUP" << std::endl;

		mouseLeftPress = false;
		break;
	}
	case cv::EVENT_RBUTTONDOWN:
		std::cout << "cv::EVENT_RBUTTONDOWN" << std::endl;

		mouseRightPress = true;
		eraserCircle(x, y, userdata);
		break;
	case cv::EVENT_RBUTTONUP:
		std::cout << "cv::EVENT_RBUTTONUP" << std::endl;

		mouseRightPress = false;
		break;
	case cv::EVENT_MBUTTONDOWN:
		std::cout << "cv::EVENT_MBUTTONDOWN" << std::endl;
		break;
	case cv::EVENT_MBUTTONUP:
		std::cout << "cv::EVENT_MBUTTONUP" << std::endl;
		break;
	case cv::EVENT_LBUTTONDBLCLK:
		std::cout << "cv::EVENT_LBUTTONDBLCLK" << std::endl;
		break;
	case cv::EVENT_RBUTTONDBLCLK:
		std::cout << "cv::EVENT_RBUTTONDBLCLK" << std::endl;
		break;
	case cv::EVENT_MBUTTONDBLCLK:
		std::cout << "cv::EVENT_MBUTTONDBLCLK" << std::endl;
		break;
		/*
		case cv::EVENT_MOUSEWHEEL:
			std::cout << "cv::EVENT_MOUSEWHEEL" << std::endl;
			break;
		case cv::EVENT_MOUSEHWHEEL:
			std::cout << "cv::EVENT_MOUSEHWHEEL" << std::endl;
			break;
		*/
	}
}



//���ڷ� ���� vector�� Mnist �Ʒ� �����͸� �Ľ��� Matrix���� ����
void OpencvPractice::MnistImageMatDataRead(std::string filePath, std::vector<cv::Mat>& vec, int readStartIndex, int readDataNum)
{
	std::ifstream file(filePath, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0; //
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		//ifstream::read(str, count)�� count��ŭ �о� str�� ����
		//char�� 1����Ʈ, int�� 4����Ʈ�̹Ƿ� int 1���� char 4���� ������ŭ ������
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		
		std::cout << readDataNum << std::endl;
		if (readDataNum > number_of_images || readDataNum <= 0)
			readDataNum = number_of_images;
		std::cout << readDataNum << std::endl;
		//startIndex�������� Ŀ���� �̵�
		//(magicnumber���� ���� ��� Ÿ�� ������ unsigned byte 1ä�� ����� ���)
		if (ConvertCVGrayImageType(magic_number) == CV_8UC1) {
			//���� Ŀ���κ��� ù��° ���ڸ�ŭ Ŀ���� �̵�
			file.seekg(sizeof(unsigned char) * readStartIndex * n_rows * n_cols, std::ios::cur);
		}
		//for (int i = 0; i < readStartIndex; ++i)
		//{
		//	for (int r = 0; r < n_rows; ++r)
		//	{
		//		for (int c = 0; c < n_cols; ++c)
		//		{
		//			//magicnumber���� ���� Ÿ�� ������ unsigned byte �� ���
		//			if (ConvertCVGrayImageType(magic_number) == CV_8UC1) {
		//				unsigned char temp = 0;
		//				file.read((char*)&temp, sizeof(temp));
		//			}
		//		}
		//	}
		//}
		
		//startIndex���� startIndex + readDataNum���� �б�
		for (int i = readStartIndex; i < readStartIndex + readDataNum; ++i)
		{
			cv::Mat tp = cv::Mat::zeros(n_rows, n_cols, ConvertCVGrayImageType(magic_number));
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					//magicnumber���� ���� ��� Ÿ�� ������ unsigned byte 1ä�� ����� ���
					if (ConvertCVGrayImageType(magic_number) == CV_8UC1) {
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						tp.at<uint8_t>(r, c) = (int)temp;
					}
				}
			}
			vec.push_back(tp);
		}
	}
}

void OpencvPractice::MnistImageLabelDataRead(std::string filePath, std::vector<uint8_t>& vec, int readStartIndex, int readDataNum)
{
	std::ifstream file(filePath, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;

		//ifstream::read(str, count)�� count��ŭ �о� str�� ����
		//char�� 1����Ʈ, int�� 4����Ʈ�̹Ƿ� int 1���� char 4���� ������ŭ ������
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		if (readDataNum > number_of_images || readDataNum <= 0)
			readDataNum = number_of_images;

		//startIndex�������� Ŀ���� �̵�
		//(magicnumber���� ���� ��� Ÿ�� ������ unsigned byte 1ä�� ����� ���)
		if (ConvertCVGrayImageType(magic_number) == CV_8UC1) {
			//���� Ŀ���κ��� ù��° ���ڸ�ŭ Ŀ���� �̵�
			file.seekg(sizeof(uint8_t) * readStartIndex, std::ios::cur);
		}

		for (int i = readStartIndex; i < readStartIndex + readDataNum; ++i)
		{
			//magicnumber���� ���� ��� Ÿ�� ������ unsigned byte 1ä�� ����� ���
			if (ConvertCVGrayImageType(magic_number) == CV_8UC1) {
				uint8_t temp = 0;
				file.read((char*)&temp, sizeof(temp));
				vec.push_back(temp);
			}
		}
	}
}

int OpencvPractice::ConvertCVGrayImageType(int magicNumber)
{
	magicNumber = (magicNumber >> 8) & 255; //3��° ����Ʈ(�ȼ� Ÿ��)�� ��������
	//��Ʋ ����� CPU���� magicNumber = ((char*)&magicNumber)[1];�� ����
	//�� ����� CPU���� magicNumber = ((char*)&magicNumber)[2];�� ����

	switch (magicNumber) {
	case 0x08: return CV_8UC1;//unsigned byte, ��� ä�� ����
	case 0x09: return CV_8SC1;//signed byte, ��� ä�� ����
	case 0x0B: return CV_16SC1;//short(2 ����Ʈ), ��� ä�� ����
	case 0x0C: return CV_32SC1;//int(4 ����Ʈ), ��� ä�� ����
	case 0x0D: return CV_32FC1;//float(4 ����Ʈ), ��� ä�� ����
	case 0x0E: return CV_64FC1;//double(8 ����Ʈ), ��� ä�� ����
	default: return CV_8UC1;
	}
}

