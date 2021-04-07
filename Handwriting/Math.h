#include "framework.h"

namespace Math {
    //Input ����� �ռ������� �ռ��� ��� ����� �� ������ ���� �е�
    void CreateZeroPadding(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& convResultSize, const cv::Size& k, const cv::Size& stride);
    
    //��� Ȯ��
    void ExpandMatrix(cv::InputArray _Input, cv::OutputArray _Output, int top, int bottom, int left, int right);

    //���� �е�, ���� ��� ����
    void Convolution(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, cv::InputArray k, const cv::Size& stride);


    //Ȱ��ȭ �Լ� Relu
    void Relu(cv::InputArray _Input, cv::OutputArray _Output);
    
    void MaxPooling(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& poolSize, const cv::Size& stride);
    
    //Ǯ�� �Լ��� �Է��� ��İ� ��� ����� ����, ������ Ǯ�� ���͸� ��ȯ
    void MaxPoolingReverse(cv::InputArray _PoolInput, cv::OutputArray _PoolFilter, cv::InputArray _PoolResult);
    
    //�Է� ��İ� K����� �ռ����� �м���, ������ K�� �̺� ����� ��ȯ
    void ConvolutionReverse(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, cv::InputArray k, const cv::Size& stride);

    void NeuralNetwork(cv::InputArray _Input, cv::OutputArray _Output, cv::InputArray w);

    //1���� �迭 ��� ������ ���� Softmax�� ��ķ� ��ȯ
    void SoftMax(cv::InputArray _Input, cv::OutputArray _Output);

   
};