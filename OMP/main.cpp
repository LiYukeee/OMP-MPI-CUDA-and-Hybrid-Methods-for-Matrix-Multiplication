#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <thread>
#include <math.h>
using namespace std;

#include "Matrix.h"
#include "OMPMatrixMul.h"

#define error 0.000001

/// <summary>
/// 单线程矩阵乘法计算方法
/// C = A × B


int main(int argc, char** argv) {
	int mmm, nnn, kkk;
	mmm = 128;
	nnn = 128;
	kkk = 128;
	if (argc >= 4) {
		mmm = atoi(argv[1]);
		nnn = atoi(argv[2]);
		kkk = atoi(argv[3]);
	}
	printf("Order of matrix: %d-%d-%d\n", mmm, nnn, kkk);
	printf("Maximum allowable error of this operation: %f\n", error);

	double start, end, time_single, time_omp;
	Matrix m_a, m_b, m_c_omp, m_c_single;
	matrixCreateAndInit(m_a, mmm, nnn);
	matrixCreateAndInit(m_b, nnn, kkk);
	matrixCreate(m_c_single, mmm, kkk);
	matrixCreate(m_c_omp, mmm, kkk);
	
	//单线程
	start = clock();
	matrixMulSingle(m_a, m_b, m_c_single);
	end = clock();
	time_single = end - start;
	cout << "Single thread runtime: \t" << time_single << "ms\t" << endl;

	//OMP
	start = clock();
	matrixMulOMP(m_a, m_b, m_c_omp);
	end = clock();
	time_omp = end - start;
	cout << "omp method: \t" << time_omp << "ms\t" <<//运行时长
		"speedUP:" << time_single / time_omp << "\t  " <<//加速比
		"maximum error: " << maxErrorOfMatrixs(m_c_single, m_c_omp) << "\t";//最大误差
	if (matrixCompare(m_c_single, m_c_omp, error)) {
		cout << "Result Correct" << endl;
	}
	else { cout << "Result Error" << endl; }
	cout << "ending..." << endl;

	//printMatrix(m_c_single);
	//printMatrix(m_c_omp);
}