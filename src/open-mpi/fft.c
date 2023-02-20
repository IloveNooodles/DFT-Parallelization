/* compile: mpicc mpi.c -o mpi */
/* run: mpirun -n 4 ./bin/parallel_mpi*/
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

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

void fft_2d(cplx buf[], int rowLen, int offset, int block_size) {
	int i;

	for(i = rowLen * offset; i < rowLen * (offset + block_size); i += rowLen) {
		fft(buf+i, rowLen);
	}
    MPI_Gather(buf + rowLen * offset, rowLen * block_size, MPI_DOUBLE_COMPLEX, buf, rowLen * block_size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	transpose(buf, rowLen);
    MPI_Bcast(buf, rowLen * rowLen, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	for(i = rowLen * offset; i < rowLen * (offset + block_size); i += rowLen) {
		fft(buf+i, rowLen);
	}
    MPI_Gather(buf + rowLen * offset, rowLen * block_size, MPI_DOUBLE_COMPLEX, buf, rowLen * block_size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	transpose(buf, rowLen);
    MPI_Bcast(buf, rowLen * rowLen, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    int world_size;
    int world_rank;

    double start, finish;

    int rowLen, offset;
    cplx mat[MAX_N * MAX_N];

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0){
        scanf("%d", &rowLen);
        for (int i = 0; i < rowLen*rowLen; i++){
            double element;
            scanf("%lf", &(element));
            mat[i] = element + 0.0I;
        }
    }

    MPI_Bcast(&(rowLen), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(mat, rowLen * rowLen, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    int block_size = rowLen / world_size;
    offset = world_rank * block_size;

    start = MPI_Wtime();

    fft_2d(mat, rowLen, offset, block_size);

    finish = MPI_Wtime();

    if (world_rank == 0){
        cplx sum = 0;
        for (int i = 0; i < rowLen*rowLen; i++){
            sum += mat[i];
        }

        sum /= rowLen*rowLen*rowLen;

        printf("Elapsed time: %e seconds\n", finish - start);
        printf("Average : (%lf, %lf)", creal(sum), cimag(sum));
    }

    MPI_Finalize();

    return 0;
}