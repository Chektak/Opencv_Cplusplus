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

#include "Math.h" 
#include "OpencvPractice.h"
#include "CNNMachine.h"

#define USEDATA_NUM 1
#define KERNEL1_NUM 2
#define KERNEL2_NUM 1
//숫자 0~9를 분류하므로 클래스 분류 수는 10개
#define CLASSIFICATIONNUM 10