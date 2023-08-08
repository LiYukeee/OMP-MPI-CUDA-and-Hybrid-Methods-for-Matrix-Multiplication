#include"CudaMatrixMulMethod.cuh"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define BLOCK_SIZE 16
#define error 0.2

int main(int argc, char** argv) {
	//矩阵尺寸
	int mmm, nnn, kkk;
	mmm = 512;
	nnn = 512;
	kkk = 512;
	if (argc >= 4) {
		mmm = atoi(argv[1]);
		nnn = atoi(argv[2]);
		kkk = atoi(argv[3]);
	}

	printf("Order of matrix: %d-%d-%d\n", mmm, nnn, kkk);
	printf("Maximum allowable error of this operation: %f\n", error);
	Matrix A, B, C_single, C_cudaV1, C_cudaV2;
	float start, end;
	float time_cuda_V1, time_cuda_V2, time_single;  //用时
	matrixCreateAndInit(A, mmm, nnn);
	matrixCreateAndInit(B, nnn, kkk);
	matrixCreate(C_single, mmm, kkk);
	matrixCreate(C_cudaV1, mmm, kkk);
	matrixCreate(C_cudaV2, mmm, kkk);

	// 单线程方法
	start = clock();
	matrixMulSingle(A, B, C_single);
	end = clock();
	time_single = end - start;
	cout << "Single thread runtime: \t" << time_single << "ms\t" << endl;

	// Cuda方法V1
	time_cuda_V1 = matrixMulCudaV1(A, B, C_cudaV1, 100);
	cout << "Cuda method 1 runtime: \t" << time_cuda_V1 << "ms\t" <<//运行时长
		"speedUP:" << time_single / time_cuda_V1 << "\t  " <<//加速比
		"maximum error: " << maxErrorOfMatrixs(C_single, C_cudaV1) << "\t";//最大误差
	if (matrixCompare(C_single, C_cudaV1, error)) {//在误差允许范围结果是否正确
		cout << "Result Correct" << endl;
	}
	else { cout << "Result Error" << endl; }

	// Cuda方法V2
	time_cuda_V2 = matrixMulCudaV2(A, B, C_cudaV2, 100);
	cout << "Cuda method 2 runtime: \t" << time_cuda_V2 << "ms\t" <<
		"speedUP:" << time_single / time_cuda_V2 << "\t  " <<
		"maximum error: " << maxErrorOfMatrixs(C_single, C_cudaV2) << "\t";
	if (matrixCompare(C_single, C_cudaV2, error)) {
		cout << "Result Correct" << endl;
	}
	else { cout << "Result Error" << endl; }

	//printMatrix(C_cudaV1);
	//printMatrix(C_cudaV2);
	cout << "ending..." << endl;
}