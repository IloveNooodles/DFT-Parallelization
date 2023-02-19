/* compile: mpicc mpi.c -o mpi */
/* run: mpirun -n 4 ./bin/parallel_mpi*/
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_N 512

struct Matrix {
    int size;
    double mat[MAX_N][MAX_N];
};

int readMatrix(int world_rank, int *n, struct Matrix *m, MPI_Comm comm) {
    if(world_rank == 0){
        scanf("%d", &(m->size));
        for (int i = 0; i < m->size; i++){
        for (int j = 0; j < m->size; j++){
            scanf("%lf", &(m->mat[i][j]));
        }
        }
    }

    MPI_Bcast(&(m->size), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(m->mat[0][0]), m->size * m->size,MPI_DOUBLE, 0, comm);
    return m->size;
}

void fft(double complex *x, int n) {
    if (n <= 1) {
        return;
    }

    double complex *even = malloc(n/2 * sizeof(double complex));
    double complex *odd = malloc(n/2 * sizeof(double complex));
    for (int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i+1];
    }

    fft(even, n/2);
    fft(odd, n/2);

    for (int i = 0; i < n/2; i++) {
        double complex w = cexp(-2.0 * M_PI * I * i / n) * odd[i];
        x[i] = even[i] + w;
        x[i+n/2] = even[i] - w;
    }

    free(even);
    free(odd);
}

void fft_2d(struct Matrix *m, int n, int world_rank, int world_size) {
    int offset = world_rank * (n/world_size);

    double complex *row = malloc(n * sizeof(double complex));
    double complex *col = malloc(n * sizeof(double complex));
    double complex *local_row = malloc((n/world_size) * sizeof(double complex));
    double complex *local_col = malloc((n/world_size) * sizeof(double complex));

    for (int i = offset; i < offset + (n/world_size); i++){
        for (int j = 0; j < n; j++){
        row[j] = m->mat[i][j];
        }
        fft(row, n);
        for (int j = 0; j < n; j++){
        m->mat[i][j] = row[j];
        }
    }

    MPI_Gather(&m->mat[offset][0], n*(n/world_size), MPI_DOUBLE, &m->mat[0][0], n*(n/world_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n/world_size; j++){
            col[i] = m->mat[offset + j][i];
        }
        MPI_Allgather(&col[offset], n/world_size, MPI_DOUBLE_COMPLEX, local_col, n/world_size, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
        for (int j = 0; j < n; j++){
            m->mat[j][offset + i] = local_col[j - (world_rank * (n/world_size))];
        }
    }

    for (int i = offset; i < offset + (n/world_size); i++){
        for (int j = 0; j < n; j++){
            col[j] = m->mat[i][j];
        }
        fft(col, n);
        for (int j = 0; j < n; j++){
            m->mat[i][j] = col[j];
        }
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n/world_size; j++){
            local_col[i] = m->mat[offset + j][i];
        }
    }
    MPI_Gather(local_row, n*(n/world_size), MPI_DOUBLE_COMPLEX, &m->mat[0][0], n*(n/world_size), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    free(row);
    free(col);
    free(local_row);
    free(local_col);
}

void printMatrix(struct Matrix *m) {
    for (int i = 0; i < m->size; i++){
        for (int j = 0; j < m->size; j++){
            printf("%.2f + %.2fi ", creal(m->mat[i][j]), cimag(m->mat[i][j]));
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int world_rank, world_size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    struct Matrix m;
    int n = readMatrix(world_rank, &n, &m, MPI_COMM_WORLD);

    fft_2d(&m, n, world_rank, world_size);

    if (world_rank == 0){
        printMatrix(&m);
    }

    MPI_Finalize();
    return 0;
}