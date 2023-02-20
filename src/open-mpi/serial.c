/* compile: mpicc mpi.c -o mpi */
/* run: mpirun -n 4 ./bin/parallel_mpi*/
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_N 512
typedef double complex cplx;

void fft(cplx buf[], int n) {
	int i, j, len;
	for (i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1)
				j ^= bit;
		j ^= bit;

		cplx temp;
        if (i < j) {
			temp = buf[i];
			buf[i] = buf[j];
			buf[j] = temp;
		}
    }

	cplx w, u, v;
    for (len = 2; len <= n; len <<= 1)  {
		double ang = 2 * M_PI / len;

		for (i = 0; i < n; i += len)  {
			for (j = 0; j < (len / 2); j++) {
				w = cexp(-I * ang * j);
				u = buf[i+j];
				v = buf[i+j+(len/2)] * w;
				buf[i+j] = u + v;
				buf[i+j+(len/2)] = u - v;
			}
		}
    }
}

void transpose(cplx buf[], int rowLen) {
	int i, j;
	cplx temp;
	for (i = 0; i < rowLen; i++) {
		for (j = i+1; j < rowLen; j++) {
			temp = buf[i*rowLen + j];
			buf[i*rowLen + j] = buf[j*rowLen + i];
			buf[j*rowLen + i] = temp;
		}
	}
}

void fft_2d(cplx buf[], int rowLen, int n) {
	int i;
	for(i = 0; i < n; i += rowLen) {
		fft(buf+i, rowLen);
	}

	transpose(buf, rowLen);

	for(i = 0; i < n; i += rowLen) {
		fft(buf+i, rowLen);
	}

	transpose(buf, rowLen);
}

int main(int argc, char** argv) {
    int rowLen;
    cplx mat[MAX_N * MAX_N];
    scanf("%d", &rowLen);
    for (int i = 0; i < rowLen*rowLen; i++){
        double element;
        scanf("%lf", &(element));
        mat[i] = element + 0.0I;
    }

    fft_2d(mat, rowLen, rowLen*rowLen);

    cplx sum = 0;

    for (int i = 0; i < rowLen*rowLen; i++){
        sum += mat[i];
    }

    sum /= rowLen*rowLen*rowLen;

    printf("Average : (%lf, %lf)", creal(sum), cimag(sum));

    return 0;
}