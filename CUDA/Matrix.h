#include <stdio.h>
#include <iostream>
using namespace std;

typedef struct {
	int height;  //矩阵行数
	int width;  //矩阵列数
	//定位矩阵M行N列元素方式：M.elements[M * M.stride + N]
	//该元素一般情况下与列数相等
	//在生成某矩阵的子矩阵之后便于定位子矩阵的元素
	int stride;
	float* elements;  //用一维数组表示矩阵中的元素
} Matrix;
/// <summary>
/// 创建矩阵,分配矩阵元素的内存空间
/// </summary>
/// <param name="M">Matrix M</param>
/// <param name="m">矩阵行数</param>
/// <param name="n">矩阵列数</param>
void matrixCreate(Matrix& M, int m, int n);
/// <summary>
/// 为矩阵中的元素随机赋值
/// </summary>
/// <param name="M">Matrix M</param>
void initMatrix(Matrix& M);
/// <summary>
/// 创建并初始化矩阵：分配内存空间并将每个元素随机赋值
/// </summary>
/// <param name="M">Matrix M</param>
/// <param name="m">矩阵行数</param>
/// <param name="n">矩阵列数</param>
void matrixCreateAndInit(Matrix& M, int m, int n);
/// <summary>
/// 打印矩阵元素
/// </summary>
/// <param name="M">Matrix M</param>
void printMatrix(Matrix& M);
/// <summary>
/// 比较两个矩阵元素是否相等
/// </summary>
/// <param name="A">Matrix A</param>
/// <param name="B">Matrix B</param>
/// <param name="er">允许的最大误差</param>
/// <returns>相等与否，相等返回True，不相等返回False</returns>
bool matrixCompare(Matrix& A, Matrix& B, float er);
/// <summary>
/// 两矩阵的最大误差
/// </summary>
/// <param name="A">Matrix A</param>
/// <param name="B">Matrix B</param>
/// <returns>最大误差</returns>
float maxErrorOfMatrixs(Matrix A, Matrix B);
/// <summary>
/// 单线程普通矩阵乘法
/// </summary>
/// <param name="A"></param>
/// <param name="B"></param>
/// <param name="C"></param>
void matrixMulSingle(Matrix& A, Matrix& B, Matrix& C);
