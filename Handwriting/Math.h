#include "framework.h"

namespace Math {
    //���� �е��� ���� ���� �е� ����
    void CreateZeroPadding(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, const cv::Size& k, const cv::Size& stride);
    
    //���� �е� ���� ��� ����
    void Convolution(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, cv::InputArray k, const cv::Size& stride);

    //Ȱ��ȭ �Լ� Relu
    void Relu(cv::InputArray _Input, cv::OutputArray _Output);
    
    void MaxPooling(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& poolSize, const cv::Size& stride);

    void NeuralNetwork(cv::InputArray _Input, cv::OutputArray _Output, cv::InputArray w);

    //1���� �迭 ��� ������ ���� Softmax�� ��ķ� ��ȯ
    void SoftMax(cv::InputArray _Input, cv::OutputArray _Output);
};