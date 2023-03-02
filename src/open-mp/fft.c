/* compile: mpicc mpi.c -o mpi */
/* run: mpirun -n 4 ./bin/parallel_mpi*/
#include <omp.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_N 512
typedef double complex cplx;

void transpose_matrix(cplx mat[], int rowLen) {
	int i, j;
	cplx temp;

	for (i = 0; i < rowLen; i++) {
		for (j = i+1; j < rowLen; j++) {
			temp = mat[i*rowLen + j];
			mat[i*rowLen + j] = mat[j*rowLen + i];
			mat[j*rowLen + i] = temp;
		}
	}
}

void fft(cplx mat[], int n) {
	int i, j, len;

    j = 0;
	for (i = 1; i < n; i++) {
		int bit = n >> 1;

		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;

		cplx temp;
        if (i < j) {
			temp = mat[i];
			mat[i] = mat[j];
			mat[j] = temp;
		}
    }

	cplx u, v;
    for (len = 2; len <= n; len <<= 1)  {
		double ang = 2 * M_PI / len;

		for (i = 0; i < n; i += len)  {
			for (j = 0; j < (len / 2); j++) {
				u = mat[i + j];
				v = mat[i + j + (len/2)] * cexp(-I * ang * j);

				mat[i + j] = u + v;
				mat[i + j + (len / 2)] = u - v;
			}
		}
    }
}

void fft_2d(cplx buf[], int rowLen, int n) {
	int i;

	for(i = 0; i < n; i += rowLen) fft(buf+i, rowLen);
	transpose_matrix(buf, rowLen);

	for(i = 0; i < n; i += rowLen) fft(buf+i, rowLen);
	transpose_matrix(buf, rowLen);
}

int main(int argc, char** argv) {
    int rowLen;
    cplx mat[MAX_N * MAX_N], sum = 0;

    scanf("%d", &rowLen);
    for (int i = 0; i < rowLen*rowLen; i++){
        double element;
        scanf("%lf", &(element));
        mat[i] = element + 0.0I;
    }

	double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            fft_2d(mat, rowLen, rowLen*rowLen);
        }
    }

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < rowLen*rowLen; i++) {
        sum += mat[i];
    }
    sum /= rowLen*rowLen*rowLen;
    double end = omp_get_wtime();

	printf("Elapsed time: %e seconds\n", (end - start) / CLOCKS_PER_SEC);
    printf("Average : (%lf, %lf)", creal(sum), cimag(sum));
    return 0;
}