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

void matrixCreate(Matrix& M, int m, int n);
void initMatrix(Matrix& M);
void matrixCreateAndInit(Matrix& M, int m, int n);
void printMatrix(Matrix& M);
bool matrixCompare(Matrix& A, Matrix& B, float er);
float maxErrorOfMatrixs(Matrix A, Matrix B);
void matrixMulSingle(Matrix& A, Matrix& B, Matrix& C);
