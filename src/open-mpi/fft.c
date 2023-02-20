/* compile: mpicc mpi.c -o mpi */
/* run: mpirun -n 4 ./bin/parallel_mpi*/
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_N 512
typedef double complex cplx;

struct Matrix {
    int size;
    cplx mat[MAX_N * MAX_N];
};

void read_matrix(struct Matrix *m, int world_rank) {
    if(world_rank == 0){
        int i;
        scanf("%d", &(m->size));
        for (i = 0; i < m->size * m->size; i++){
            double element;
            scanf("%lf", &(element));
            m->mat[i] = element + 0.0I;
        }
    }

    MPI_Bcast(&(m->size), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m->mat, m->size * m->size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}

void print_result(struct Matrix *m, int world_rank, double elapsed_time) {
    if (world_rank != 0) return;

    cplx sum = 0;
    int i;

    for (i = 0; i < m->size * m->size; i++) sum += m->mat[i];
    sum /= m->size * m->size * m->size;

    printf("Elapsed time: %e seconds\n", elapsed_time);
    printf("Average : (%lf, %lf)", creal(sum), cimag(sum));
}

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

void fft_2d(cplx buf[], int rowLen, int world_rank, int world_size) {
	int i;
    int block_size = rowLen / world_size;
    int offset = world_rank * block_size;

	for(i = rowLen * offset; i < rowLen * (offset + block_size); i += rowLen) fft(buf+i, rowLen);
    MPI_Gather(buf + rowLen * offset, rowLen * block_size, MPI_DOUBLE_COMPLEX, buf, rowLen * block_size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	transpose(buf, rowLen);
    MPI_Bcast(buf, rowLen * rowLen, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	for(i = rowLen * offset; i < rowLen * (offset + block_size); i += rowLen) fft(buf+i, rowLen);
    MPI_Gather(buf + rowLen * offset, rowLen * block_size, MPI_DOUBLE_COMPLEX, buf, rowLen * block_size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

	transpose(buf, rowLen);
    MPI_Bcast(buf, rowLen * rowLen, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    int world_size, world_rank;
    double start_time, finish_time;
    struct Matrix m;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    read_matrix(&m, world_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    start_time = MPI_Wtime();
    fft_2d(m.mat, m.size, world_rank, world_size);
    finish_time = MPI_Wtime();

    print_result(&m, world_rank, finish_time - start_time);
    MPI_Finalize();
    return 0;
}