#pragma once
#include "framework.h"

namespace Layers
{
	template <typename T>
	void A(T a) {
		a.cry();
	}

	template <typename ForwardParam, typename ForwardReturn, typename BackwardParam, typename BackwardReturn>
	class Layer{
	protected :

	public :
		Layer(){
			
		}
		//레이어 내부 변수 초기화 (e.g : 커널 사이즈, 패딩 옵션, 가중치 행렬, 편향)
		virtual void Initialize() = 0;
		template <typename ForwardParam, typename ForwardReturn>
		ForwardReturn Forward(ForwardParam tuple) = 0;
		
		template <typename BackwardParam, typename BackwardReturn>
		BackwardReturn Backward(BackwardParam tuple)=0;
		
	};

	//class Relu : public Layer<int> {
	//	/*template <typename ForwardParam, typename ForwardReturn>
	//	ForwardReturn Forward(ForwardParam tuple) override {};

	//	template <typename BackwardParam, typename BackwardReturn>
	//	BackwardReturn Backward(BackwardParam tuple) override {};*/
	//public :
	//	void cry() {
	//		std::cout << "렐루 ;: 이게 왜 동작하지?" << std::endl;
	//	}

	//	template<typename Type, typename ... Types>
	//	void foo(Type arg,Types... args) {

	//	}
	//};
	//
	//class Affine : public Layer {
	//	/*template <typename ForwardParam, typename ForwardReturn>
	//	ForwardReturn Forward(ForwardParam tuple) override {};

	//	template <typename BackwardParam, typename BackwardReturn>
	//	BackwardReturn Backward(BackwardParam tuple) override {};*/
	//public :
	//	void cry() {
	//		std::cout << "어파인 ;: 이게 왜 동작하지?" << std::endl;
	//	}

	//	template<typename Type, typename ... Types>
	//	void foo(Type arg,Types... args) {

	//	}
	//};
};

