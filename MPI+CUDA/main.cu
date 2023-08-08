//MPIv2 + CUDAv2方法
#include <mpi.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <thread>
#include <math.h>

using namespace std;
#define BLOCK_SIZE 16
#define error 0.2
#include"cudaMatrixMul.cuh"
#include <chrono>


//void printTime() {
//	auto now = std::chrono::system_clock::now();
//	//通过不同精度获取相差的毫秒数
//	uint64_t dis_millseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
//		- std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count() * 1000;
//	time_t tt = std::chrono::system_clock::to_time_t(now);
//	auto time_tm = localtime(&tt);
//	char strTime[25] = { 0 };
//	sprintf(strTime, "%d-%02d-%02d %02d:%02d:%02d %03d", time_tm->tm_year + 1900,
//		time_tm->tm_mon + 1, time_tm->tm_mday, time_tm->tm_hour,
//		time_tm->tm_min, time_tm->tm_sec, (int)dis_millseconds);
//	cout << strTime << endl;
//}


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
	if (argc == 2) {
		mmm = atoi(argv[1]);
		nnn = atoi(argv[1]);
		kkk = atoi(argv[1]);
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
		printf("num procs: %d\n", numprocs);
		// 将矩阵A和B随机赋值
		initMatrix(A);
		initMatrix(B);

		double start = MPI_Wtime(), stop, time_MPI, s1, s2, tn;
		// 将矩阵B发送给其他进程
		// 1 数据广播
		s1 = MPI_Wtime();
		MPI_Bcast(B.elements, B.height * B.width, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		s2 = MPI_Wtime();
		cout << "数据广播用时 :" << (s2 - s1) * 1000 << endl;

		// 将矩阵A的各行发送给各个从进程
		// 2 数据分发
		s1 = MPI_Wtime();
		MPI_Scatter(A.elements, line * A.width, MPI_FLOAT, buffer.elements, line * nnn, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		s2 = MPI_Wtime();
		cout << "数据分发用时:" << (s2 - s1) * 1000 << endl;

		// 3 计算本地结果
		s1 = MPI_Wtime();
		matrixMulCudaV2(buffer, B, ans, 1, 0);
		MPI_Barrier(MPI_COMM_WORLD);
		s2 = MPI_Wtime();
		printf("计算用时:%f\n", (s2-s1)*1000);

		// 4 结果聚集
		s1 = MPI_Wtime();
		MPI_Gather(ans.elements, line * kkk, MPI_FLOAT, C.elements, line * kkk, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		s2 = MPI_Wtime();
		printf("结果聚集用时:%f\n", (s2 - s1) * 1000);

		// 5 剩余行处理（处理不能整除的情况）
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
		// 一般来大规模矩阵相乘计算时间较长，方阵阶数到2000以上时会取消单线程结果验证
		double time_single = 0.0;
		Matrix C_single;
		matrixCreate(C_single, mmm, kkk);
		start = MPI_Wtime();
		matrixMulSingle(A, B, C_single);
		stop = MPI_Wtime();
		time_single = (stop - start);

		// 输出结果
		cout << "--single method runtime: \t " << time_single * 1000 << "ms" << endl;
		cout << "--MPIv2 + CUDAv2  runtime: \t" << time_MPI * 1000 << "ms\t" <<
			"speedUP:" << time_single / time_MPI << "\t  " <<
			"maximum error: " << maxErrorOfMatrixs(C_single, C) << "\t" << endl;

		/*printMatrix(C_single);
		printMatrix(C);*/

	}
	// 从进程接收数据，计算结果并发送给主进程
	else {
		// 数据广播
		MPI_Bcast(B.elements, B.width * B.height, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		// 数据分发
		MPI_Scatter(A.elements, line * nnn, MPI_FLOAT, buffer.elements, line * nnn, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		// 计算本地
		matrixMulCudaV2(buffer, B, ans, 1, 0);
		MPI_Barrier(MPI_COMM_WORLD);

		// 结果聚集
		MPI_Gather(ans.elements, line * kkk, MPI_FLOAT, C.elements, line * kkk, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();//结束
	return 0;
}