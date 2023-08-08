#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

#include "Matrix.h"

/// <summary>
/// 普通矩阵乘法
/// </summary>
void matrixMUL(float* A, float* B, float* C, int m, int n, int k) {
	float temp;
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			temp = 0.0f;
			for (int l = 0; l < n; ++l)
			{
				temp += A[i * n + l] * B[l * k + j];
			}
			C[i * k + j] = temp;
		}
	}
}

//MPI_Scatter和MPI_Gather实现
int main(int argc, char** argv) {
	int mmm, nnn, kkk;
	mmm = 1024;
	nnn = 1024;
	kkk = 1024;
	if (argc >= 4) {
		mmm = atoi(argv[1]);
		nnn = atoi(argv[2]);
		kkk = atoi(argv[3]);
	}
	// MPI初始化
	int rank = 0, numprocs = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //当前进程号
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);  //进程个数
	MPI_Status status;

	// 初始化数据
	Matrix A, B, C, buffer, ans;
	matrixCreate(A, mmm, nnn);
	matrixCreate(B, nnn, kkk);
	matrixCreate(C, mmm, kkk);

	int line = mmm / numprocs;
	// 缓存大小大于等于要处理的数据大小，大于时只需关注实际数据那部分
	matrixCreate(buffer, line, nnn);
	matrixCreate(ans, line, kkk);

	int i, j, k;
	// 主处理器
	if (rank == 0) {
		printf("order of matrix: %d-%d-%d\n", mmm, nnn, kkk);
		// 将矩阵A和B随机赋值
		initMatrix(A);
		initMatrix(B);

		double start = MPI_Wtime(), stop, time_MPI;
		// 将矩阵B发送给其他进程
		//数据广播
		MPI_Bcast(B.elements, B.height * B.width, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// 将矩阵A的各行发送给各个从进程
		//数据分发
		MPI_Scatter(A.elements, line * A.width, MPI_FLOAT, buffer.elements, line * nnn, MPI_FLOAT, 0, MPI_COMM_WORLD);

		//计算本地结果
		matrixMUL(buffer.elements, B.elements, ans.elements, buffer.height, buffer.width, B.width);

		//结果聚集
		MPI_Gather(ans.elements, line * kkk, MPI_FLOAT, C.elements, line * kkk, MPI_FLOAT, 0, MPI_COMM_WORLD);

		//剩余行处理（处理不能整除的情况）
		float temp;
		int rest = mmm % numprocs;
		if (rest != 0) {
			cout << "进行剩余行处理，剩余行：" << rest << endl;
			for (i = mmm - rest; i < mmm; i++)
				for (j = 0; j < kkk; j++) {
					temp = 0.0f;
					for (k = 0; k < nnn; k++)
						temp += A.elements[i * nnn + k] * B.elements[k * kkk + j];
					C.elements[i * kkk + j] = temp;
				}
		}

		stop = MPI_Wtime();
		time_MPI = stop - start;


		// 依据单线程计算结果为标准验证MPI计算结果是否正确
		double time_single;
		Matrix C_single;
		matrixCreate(C_single, mmm, kkk);
		start = MPI_Wtime();
		matrixMulSingle(A, B, C_single);
		stop = MPI_Wtime();
		time_single = stop - start;

		cout << "single method runtime: \t " << time_single << "s" << endl;
		cout << "MPI method v2 runtime: \t" << time_MPI << "s\t" <<
			"speedUP:" << time_single / time_MPI << "\t  " <<
			"maximum error: " << maxErrorOfMatrixs(C_single, C) << "\t" <<
			"Efficiency: " << time_single / time_MPI / numprocs << endl;
		//printMatrix(C_single);
		//printMatrix(C);
	}
	// 从进程接收数据，计算结果并发送给主进程
	else {
		MPI_Bcast(B.elements, B.width * B.height, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Scatter(A.elements, line * nnn, MPI_FLOAT, buffer.elements, line * nnn, MPI_FLOAT, 0, MPI_COMM_WORLD);
		// 计算本地
		matrixMUL(buffer.elements, B.elements, ans.elements, buffer.height, buffer.width, B.width);
		// 结果聚集
		MPI_Gather(ans.elements, line * kkk, MPI_FLOAT, C.elements, line * kkk, MPI_FLOAT, 0, MPI_COMM_WORLD);

	}

	MPI_Finalize();//结束
	return 0;
}