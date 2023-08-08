#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Matrix.h"
#define BLOCK_SIZE 16

int calculateDimGrid(int a, int b) {
	float A = float(a);
	float B = float(b);
	float C = (A + (B - 1)) / B;
	return int(C);
}

/// <summary>
/// 计算核心：直接计算版本
/// </summary>
__global__ void MatMulKernelV1(Matrix A, Matrix B, Matrix C)
{
	int m = A.height;
	int n = A.width;
	int k = B.width;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;
	if (col < k && row < m)
	{
		for (int i = 0; i < n; i++)
		{
			sum += A.elements[row * n + i] * B.elements[i * k + col];
		}
		C.elements[row * k + col] = sum;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// CUDA矩阵乘法：普通方法
/// C = A × B
/// </summary>
/// <param name="A">矩阵类A</param>
/// <param name="B">矩阵类B</param>
/// <param name="C">矩阵类C</param>
/// <param name="times">计算次数</param>
/// <param name="warmupTimes">预热次数</param>
/// <returns>单次内核运行时间（单位：毫秒）</returns>
float matrixMulCudaV1(const Matrix A, const Matrix B, Matrix C, int times, int warmupTimes = 10)
{
	float start, end, diffTime;
	//加载矩阵到显存
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height; d_A.stride = A.stride;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
		cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height; d_B.stride = B.stride;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,
		cudaMemcpyHostToDevice);

	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height; d_C.stride = C.stride;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	//启动核
	cudaThreadSynchronize();
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	dim3 dimGrid(calculateDimGrid(B.width, dimBlock.x), calculateDimGrid(A.height, dimBlock.y));
	start = clock();
	for (int num = 0; num < warmupTimes; ++num) {
		MatMulKernelV1 << <dimGrid, dimBlock >> > (d_A, d_B, d_C);
		cudaDeviceSynchronize();
	}//预启动，获得更加精确的计时
	for (int num = 0; num < 100; ++num) {
		MatMulKernelV1 << <dimGrid, dimBlock >> > (d_A, d_B, d_C);
		cudaDeviceSynchronize();//CPU等待GPU任务运行完毕
	}//运行N次，为获得精确计时
	end = clock();
	//加载矩阵C到内存
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	diffTime = end - start;
	//printMatrix(C);
	//释放空间
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	return diffTime / float(times);
}

/// <summary>
/// 得到矩阵元素
/// </summary>
/// <param name="A">Matrix A</param>
/// <param name="row">行号</param>
/// <param name="col">列号</param>
/// <returns>返回改位置浮点值</returns>
__device__ float GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}
/// <summary>
/// 更改矩阵元素
/// </summary>
/// <param name="A">Matrix A</param>
/// <param name="row">行号</param>
/// <param name="col">列号</param>
/// <param name="value">修改值</param>
__device__ void SetElement(Matrix A, int row, int col, float value)
{
	A.elements[row * A.stride + col] = value;
}
/// <summary>
/// 得到子矩阵
/// </summary>
/// <param name="A">母矩阵A</param>
/// <param name="row">行号</param>
/// <param name="col">列号</param>
/// <returns>Matrix 子矩阵</returns>
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}
__device__ int calculateDimGridDevice(int a, int b) {
	float A = float(a);
	float B = float(b);
	float C = (A + (B - 1)) / B;
	return int(C);
}
/// <summary>
/// 计算核心：共享内存版本
/// </summary>
__global__ void MatMulKernelV2(Matrix A, Matrix B, Matrix C)
{
	// 线程块的行和列
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	// 每个线程块计算C的一块
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
	// 线程块中的每个线程计算该线程块中的一个元素
	float Cvalue = 0.0f;  //该元素的值
	// 元素在线程块中的行和列
	int row = threadIdx.y;
	int col = threadIdx.x;
	//在C矩阵中的位置
	int row_all = blockRow * BLOCK_SIZE + row;
	int col_all = blockCol * BLOCK_SIZE + col;
	//设定共享内存
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
	for (int m = 0; m < calculateDimGridDevice(A.width, BLOCK_SIZE); ++m) {
		//Matrix Asub = GetSubMatrix(A, blockRow, m);
		//Matrix Bsub = GetSubMatrix(B, m, blockCol);		
		//线程块中的每个线程读取一个元素
		if (m * BLOCK_SIZE + col < A.width && row_all < A.height)
			//As[row][col] = GetElement(Asub, row, col);
			As[row][col] = A.elements[row_all * A.width + m * BLOCK_SIZE + col];
		else
			As[row][col] = 0.0f;

		if (m * BLOCK_SIZE + row < B.height && col_all < B.width)
			//Bs[row][col] = GetElement(Asub, row, col);
			Bs[row][col] = B.elements[(m * BLOCK_SIZE + row) * B.width + col_all];
		else
			Bs[row][col] = 0.0f;
		//线程同步，等待所有线程完成读取操作
		__syncthreads();
		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];
		//在进行下一步骤之前，确保每个线程都完成了之前的操作
		__syncthreads();
		//将计算得出的值存储到全局内存中
		if (row_all < C.height && col_all < C.width)
			SetElement(Csub, row, col, Cvalue);
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// CUDA矩阵乘法：共享内存方法
/// C = A × B
/// </summary>
/// <param name="A">矩阵类A</param>
/// <param name="B">矩阵类B</param>
/// <param name="C">矩阵类C</param>
/// <param name="times">计算次数</param>
/// <param name="warmupTimes">预热次数</param>
/// <returns>单次运行时间（单位：毫秒）</returns>
float matrixMulCudaV2(const Matrix A, const Matrix B, Matrix C, int times, int warmupTimes = 10)
{
	float start, end, diffTime;
	//加载矩阵到显存
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height; d_A.stride = A.stride;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
		cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height; d_B.stride = B.stride;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,
		cudaMemcpyHostToDevice);

	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height; d_C.stride = C.stride;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	dim3 dimGrid(calculateDimGrid(B.width, dimBlock.x), calculateDimGrid(A.height, dimBlock.y));
	for (int num = 0; num < warmupTimes; ++num) {
		MatMulKernelV2 << <dimGrid, dimBlock >> > (d_A, d_B, d_C);
		cudaDeviceSynchronize();
	}//预启动，获得更加精确的计时
	start = clock();
	for (int num = 0; num < times; ++num) {
		MatMulKernelV2 << <dimGrid, dimBlock >> > (d_A, d_B, d_C);
		cudaDeviceSynchronize();
	}
	end = clock();
	//加载矩阵C到内存
	cudaMemcpy(C.elements, d_C.elements, size,
		cudaMemcpyDeviceToHost);

	diffTime = end - start;
	//printMatrix(C);
	//释放空间
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	return diffTime / float(times);
}