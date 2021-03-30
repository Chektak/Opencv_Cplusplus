#include "framework.h"

namespace Math {
    //세임 패딩을 위한 제로 패딩 생성
    void CreateZeroPadding(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, const cv::Size& k, const cv::Size& stride);
    
    //세임 패딩 교차 상관 연산
    void Convolution(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& outputSize, cv::InputArray k, const cv::Size& stride);

    //활성화 함수 Relu
    void Relu(cv::InputArray _Input, cv::OutputArray _Output);
    
    void MaxPooling(cv::InputArray _Input, cv::OutputArray _Output, const cv::Size& poolSize, const cv::Size& stride);

    void NeuralNetwork(cv::InputArray _Input, cv::OutputArray _Output, cv::InputArray w);

    //1차원 배열 요소 각각에 대한 Softmax를 행렬로 반환
    void SoftMax(cv::InputArray _Input, cv::OutputArray _Output);
};