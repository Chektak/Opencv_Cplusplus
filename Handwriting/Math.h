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
    
    //풀링 함수에 입력한 행렬과 결과 행렬을 비교해, 역으로 풀링 필터를 반환
    void MaxPoolingReverse(cv::InputArray _PoolInput, cv::OutputArray _PoolFilter, cv::InputArray _PoolResult);
    
    //입력 행렬과 K행렬의 합성곱을 분석해, 역으로 K의 미분 행렬을 반환
    void ConvolutionReverse(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, cv::InputArray k, const cv::Size& stride);

    void NeuralNetwork(cv::InputArray _Input, cv::OutputArray _Output, cv::InputArray w);

    //1차원 배열 요소 각각에 대한 Softmax를 행렬로 반환
    void SoftMax(cv::InputArray _Input, cv::OutputArray _Output);

   
};