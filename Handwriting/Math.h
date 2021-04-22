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
    
    // Ǯ�� �Լ��� �Է� ��Ŀ� ���� �̺��� ��ȯ
    void GetMaxPoolingFilter(cv::InputArray _PoolInput, cv::OutputArray _PoolFilter, cv::InputArray _PoolResult, const cv::Size& poolSize, const cv::Size& stride);
    
    //Input����� UpSampleling�� PoolFilter ��İ� ���Ͱ��� ��ȯ
    void MaxPoolingBackprop(cv::InputArray _Input, cv::OutputArray _Output,cv::InputArray _PoolFilter, const cv::Size& poolSize, const cv::Size& stride);
    
    //�ռ������� ���� �е����� ���� �Է� ����� �����е��� ��, Ŀ�� �������� �Է� ����� Start�� End �ε����� pair Output���� ��ȯ
    void GetConvBackpropFilters(cv::InputArray _Input, std::vector<std::pair<int, int>>* _Output, cv::InputArray k, const cv::Size& stride);
    
    //�ռ��� ���͸� Ȱ���� Ŀ�� ��Ŀ� �����ϴ� (Input ��� ��� * �ռ��� �Է� ��� ���)�� ���� �� ��ȯ
    void ConvKBackprop(cv::InputArray _Input, cv::InputArray _ConvZeroPadInput,cv::InputArray _Kernel, cv::OutputArray _Output, const std::vector<std::pair<int, int>>& _ConvFilter, const cv::Size& stride,float learningRate);

    //�ռ��� ���͸� Ȱ���� �ռ��� �Է� ��� ��ҿ� �����ϴ� Ŀ�� ��� ��Ҹ� ���� ���ϰ�, ��ķ� ���ø��� Input ��İ� ���Ͱ� �� ��ȯ
    void ConvXBackprop(cv::InputArray _Input, cv::InputArray _Kernel, cv::OutputArray _Output, const std::vector<std::pair<int, int>>& _ConvFilter, const cv::Size& stride,float learningRate);
    

    void NeuralNetwork(cv::InputArray _Input, cv::OutputArray _Output, cv::InputArray w);

    //1���� �迭 ��� ������ ���� Softmax�� ��ķ� ��ȯ
    void SoftMax(cv::InputArray _Input, cv::OutputArray _Output);

   
};