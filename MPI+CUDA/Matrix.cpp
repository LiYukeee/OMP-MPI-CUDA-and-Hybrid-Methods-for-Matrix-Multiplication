#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <thread>
#include <math.h>
using namespace std;

#include "Matrix.h"


/// <summary>
/// 创建矩阵,分配矩阵元素的内存空间
/// </summary>
/// <param name="M">Matrix M</param>
/// <param name="m">矩阵行数</param>
/// <param name="n">矩阵列数</param>
void matrixCreate(Matrix& M, int m, int n) {
	M.height = m;
	M.width = n;
	M.stride = M.width;
	M.elements = new float[M.height * M.width];
}
/// <summary>
/// 为矩阵中的元素随机赋值
/// </summary>
/// <param name="M">Matrix M</param>
void initMatrix(Matrix& M) {
	int m = M.height;
	int n = M.width;
	srand((unsigned int)time(NULL));
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			M.elements[i * n + j] = float((rand() % 10000) - (rand() % 10000)) / 100.0;  //rand()生成0-无穷大的随机整数
		}
	}
}
/// <summary>
/// 创建并初始化矩阵：分配内存空间并将每个元素随机赋值
/// </summary>
/// <param name="M">Matrix M</param>
/// <param name="m">矩阵行数</param>
/// <param name="n">矩阵列数</param>
void matrixCreateAndInit(Matrix& M, int m, int n) {
	matrixCreate(M, m, n);
	initMatrix(M);
}
/// <summary>
/// 打印矩阵元素
/// </summary>
/// <param name="M">Matrix M</param>
void printMatrix(Matrix& M) {
	int m = M.height;
	int n = M.width;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%12.3f\t", M.elements[i * n + j]);
		}
		cout << endl;
	}
	cout << endl;
}
///// <summary>
///// 基于OMP并行计算：比较两个矩阵元素是否相等
///// </summary>
///// <param name="A">Matrix A</param>
///// <param name="B">Matrix B</param>
///// <param name="er">允许的最大误差</param>
///// <returns>相等与否，相等返回True，不相等返回False</returns>
//bool matrixCompare(Matrix& A, Matrix& B, float er) {
//	if (A.height != B.height || A.width != B.width)
//		return false;
//
//	int m = A.height;
//	int n = A.width;
//	const int num_threads = (int)std::thread::hardware_concurrency();
//	int* res = new int[num_threads];//用于保存结果
//	for (int i = 0; i < num_threads; i++)
//		res[i] = 1;
//	omp_set_num_threads(num_threads);
//#pragma omp parallel
//	{
//		int id = omp_get_thread_num();
//		for (int i = id; i < m * n; i += num_threads) {
//			if (abs(A.elements[i] - B.elements[i]) > er) {
//				res[id] = 0;
//				break;
//			}
//		}
//	}
//	//结果汇总
//	for (int i = 0; i < num_threads; i++) {
//		if (res[i] == 0)
//			return false;
//	}
//	return true;
//}
/// <summary>
/// 两矩阵的最大误差
/// </summary>
/// <param name="A">Matrix A</param>
/// <param name="B">Matrix B</param>
/// <returns>最大误差</returns>
/// <returns>最大误差</returns>
float maxErrorOfMatrixs(Matrix A, Matrix B) {

	float maxError = 0.0f;
	int length = A.height * A.width;
	float temp;
	for (int i = 0; i < length; i++) {
		temp = abs(A.elements[i] - B.elements[i]);
		if (temp > maxError) {
			maxError = temp;
		}
	}
	return maxError;
}
/// <summary>
/// 单线程普通矩阵乘法
/// </summary>
/// <param name="A"></param>
/// <param name="B"></param>
/// <param name="C"></param>
void matrixMulSingle(Matrix& A, Matrix& B, Matrix& C) {
	int a = A.height;
	int b = A.width;
	int c = B.width;
	float temp;
	for (int i = 0; i < a; i++)
	{
		for (int j = 0; j < c; j++)
		{
			temp = 0.0f;
			for (int k = 0; k < b; k++)
			{
				temp += A.elements[i * b + k] * B.elements[k * c + j];
			}
			C.elements[i * c + j] = temp;
		}
	}
}