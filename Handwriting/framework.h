#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/core/fast_math.hpp"

//����׿�
#include <bitset>

#include <vector>
#include <iostream>
#include <fstream>
#include <random>
//#include <time.h>

#include "Math.h" 
#include "OpencvPractice.h"
#include "CNNMachine.h"

#define USEDATA_NUM 4
#define KERNEL1_NUM 1
#define KERNEL2_NUM 3
//���� 0~9�� �з��ϹǷ� Ŭ���� �з� ���� 10��
#define CLASSIFICATIONNUM 10