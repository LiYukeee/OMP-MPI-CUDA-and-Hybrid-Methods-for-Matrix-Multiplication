#include <stdio.h>
#include <iostream>
using namespace std;

#include "Matrix.h"
#include "OMPMatrixMul.h"
#include <omp.h>

void matrixMulOMP(Matrix& A, Matrix& B, Matrix& C) {
	int m = A.height;
	int n = A.width;
	int k = B.width;
	float* col;
#pragma omp parallel for schedule(dynamic) private(col)
	for (int jj = 0; jj < k; ++jj)
	{
		float sum = 0.0f;
		int id = omp_get_thread_num();
		//printf("id:%d,BµÚ%dÁÐ\n", id, jj+1);

		col = (float*)calloc(n, sizeof(float));

		for (int temp = 0; temp < n; ++temp)
			col[temp] = B.elements[temp * k + jj];

		for (int ii = 0; ii < m; ++ii) {
			sum = 0.0f;
			for (int kk = 0; kk < n; ++kk) {
				sum += col[kk] * A.elements[ii * n + kk];
			}
			C.elements[ii * k + jj] = sum;
		}
		free(col);
	}
}