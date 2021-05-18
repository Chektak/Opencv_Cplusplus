#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/core/fast_math.hpp"

//디버그용
#include <bitset>

#include <vector>
#include <iostream>
#include <fstream>
#include <random>
//#include <time.h>

//#define USEDATA_NUM 3000
//#define KERNEL1_NUM 8
//#define KERNEL2_NUM 16
//숫자 0~9를 분류하므로 클래스 분류 수는 10개
#define CLASSIFICATIONNUM 10

#include "Math.h" 
class CNNMachine;
class OpencvPractice;
#include "OpencvPractice.h"
#include "CNNMachine.h"