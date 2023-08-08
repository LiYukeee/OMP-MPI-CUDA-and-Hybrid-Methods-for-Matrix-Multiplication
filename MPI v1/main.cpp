#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

#include "Matrix.h"

//MPI_Send和MPI_Recv实现
int main(int argc, char** argv) {
	//矩阵尺寸
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
		for (i = 1; i < numprocs; ++i) {
			MPI_Send(B.elements, B.height * B.width, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		}

		// 将矩阵A的各行发送给各个从进程
		for (i = 1; i < numprocs; ++i) {
			MPI_Send(A.elements + (i - 1) * line * nnn, line * nnn, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
		}

		// 接受从进程的计算结果
		for (k = 1; k < numprocs; ++k) {
			MPI_Recv(ans.elements, line * kkk, MPI_FLOAT, k, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
			//将结果存入矩阵C
			for (i = 0; i < line; ++i) {
				for (j = 0; j < kkk; ++j) {
					C.elements[((k - 1) * line + i) * kkk + j] = ans.elements[i * kkk + j];
				}
			}
		}

		//计算A剩下的数据
		float temp;
		for (i = (numprocs - 1) * line; i < mmm; ++i) {
			for (j = 0; j < kkk; ++j) {
				temp = 0.0f;
				for (k = 0; k < nnn; ++k) {
					temp += A.elements[i * nnn + k] * B.elements[k * kkk + j];
				}
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
		cout << "MPI method v1 runtime: \t" << time_MPI << "s\t" <<
			"speedUP:" << time_single / time_MPI << "\t  " <<
			"maximum error: " << maxErrorOfMatrixs(C_single, C) << "\t" <<
			"Efficiency: " << time_single / time_MPI / numprocs << endl;
		/*printMatrix(A);
		printMatrix(B);
		printMatrix(C);*/
	}
	// 从进程接收数据，计算结果并发送给主进程
	else {
		MPI_Recv(B.elements, B.width * B.height, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(buffer.elements, line * nnn, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// 计算结果
		for (i = 0; i < line; ++i) {
			for (j = 0; j < kkk; ++j) {
				float temp = 0.0f;
				for (k = 0; k < nnn; ++k) {
					temp += buffer.elements[i * nnn + k] * B.elements[k * kkk + j];
				}
				ans.elements[i * kkk + j] = temp;
			}
		}
		// 将结果发送给主进程
		MPI_Send(ans.elements, line * kkk, MPI_INT, 0, 3, MPI_COMM_WORLD);
	}

	MPI_Finalize();//结束
	return 0;
}