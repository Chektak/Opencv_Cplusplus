#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/core/fast_math.hpp"

#include <vector>
#include <bitset>
#include <iostream>
#include <fstream>
#include <random>
//#include <time.h>

#include "Math.h" 
#include "OpencvPractice.h"

#define USEDATA_NUM 60000
#define KERNEL1_NUM 32
#define KERNEL2_NUM 64
//숫자 0~9를 분류하므로 클래스 분류 수는 10개
#define CLASSIFICATIONNUM 10
