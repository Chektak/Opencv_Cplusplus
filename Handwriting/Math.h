#include "framework.h"

namespace Math {
    //Input 행렬이 합성곱으로 합성곱 결과 사이즈가 될 때까지 제로 패딩
    void CreateZeroPadding(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& convResultSize, const cv::Size& k, const cv::Size& stride);
    
    //행렬 확장
    void ExpandMatrix(cv::InputArray _Input, cv::OutputArray _Output, int top, int bottom, int left, int right);

    //세임 패딩, 교차 상관 연산
    void Convolution(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, cv::InputArray k, const cv::Size& stride);


    //활성화 함수 Relu
    void Relu(cv::InputArray _Input, cv::OutputArray _Output);
    
    void MaxPooling(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& poolSize, const cv::Size& stride);
    
    // 풀링 함수의 입력 행렬에 대한 미분을 반환
    void GetMaxPoolingFilter(cv::InputArray _PoolInput, cv::OutputArray _PoolFilter, cv::InputArray _PoolResult, const cv::Size& poolSize, const cv::Size& stride);
    
    //Input행렬을 UpSampleling해 PoolFilter 행렬과 벡터곱해 반환
    void MaxPoolingBackprop(cv::InputArray _Input, cv::OutputArray _Output,cv::InputArray _PoolFilter, const cv::Size& poolSize, const cv::Size& stride);
    
    //합성곱에서 제로 패딩되지 않은 입력 행렬이 제로패딩될 때, 커널 기준으로 입력 행렬의 Start와 End 인덱스를 pair Output으로 반환
    void GetConvBackpropFilters(cv::InputArray _Input, std::vector<std::pair<int, int>>* _Output, cv::InputArray k, const cv::Size& stride);
    
    //합성곱 필터를 활용해 커널 행렬에 대응하는 (Input 행렬 요소 * 합성곱 입력 행렬 요소)를 더하고, 커널 크기의 행렬로 샘플링해 반환
    void ConvKBackprop(cv::InputArray _Input, cv::InputArray _ConvZeroPadInput,const cv::Size kernelMatSize, cv::OutputArray _Output, const std::vector<std::pair<int, int>>& _ConvFilter, const cv::Size& stride);

    //합성곱 필터를 활용해 합성곱 입력 행렬 요소에 대응하는 커널 행렬 요소를 전부 더하고, 합성곱 입력 행렬 크기의 행렬로 샘플링해 반환
    void ConvXBackprop(cv::InputArray _Input, cv::InputArray _Kernel, cv::OutputArray _Output, const std::vector<std::pair<int, int>>& _ConvFilter, const cv::Size& stride);
    

    void NeuralNetwork(cv::InputArray _Input, cv::OutputArray _Output, cv::InputArray w);

    //1차원 배열 요소 각각에 대한 Softmax를 행렬로 반환
    void SoftMax(cv::InputArray _Input, cv::OutputArray _Output);

    //절댓값 반환
    double Absolute(double value);

    void OneHotEncoding(cv::InputArray _Input, cv::OutputArray _Output);
    
    double Clip(double min, double max, const double& _InputValue);
};