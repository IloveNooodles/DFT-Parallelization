/* compile: mpicc mpi.c -o mpi */
/* run: mpirun -n 4 ./bin/parallel_mpi*/
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_N 512

struct Matrix
{
  int size;
  double mat[MAX_N][MAX_N];
};

struct FreqMatrix
{
  int size;
  double complex mat[MAX_N][MAX_N];
};

int readMatrix(int world_rank, int *n, struct Matrix *m, MPI_Comm comm)
{
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

double complex dft(struct Matrix *mat, int k, int l)
{
  double complex element = 0.0;
  for (int m = 0; m < mat->size; m++)
  {
    for (int n = 0; n < mat->size; n++)
    {
      double complex arg = (k * m / (double)mat->size) + (l * n / (double)mat->size);
      double complex exponent = cexp(-2.0I * M_PI * arg);
      element += mat->mat[m][n] * exponent;
    }
  }
  return element / (double)(mat->size * mat->size);
}

int main(void)
{
  /* global variable */
  struct Matrix source;

  int world_size;
  int world_rank;
  
  int n, k, l, i;
  double elapsed;
  double complex sum = 0;

  /* local variable */
  int offset;
  double start, finish, loc_elapsed;
  double complex local_sum;

  /* init the MPI process */
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  /* read matrix from stdin and set size iff worldrank 0 */
  n = readMatrix(world_rank, &n, &source, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  /* get start time */
  start = MPI_Wtime();

  /* block_size will be block_size * block_size; */
  int block_size = n / world_size; 
  int iteration = n * n / (block_size * block_size * world_size);
  offset = world_rank * block_size; // 0 * 16, 1 * 16
  
  /* will compute block_size * block_size iteration times each process */
  local_sum = 0.0;
  for(i = 0; i < iteration; i++){
    for(k = i * block_size; k < block_size * (i + 1); k++){
      for(l = 0; l < block_size; l++){
        local_sum += dft(&source, k, l + offset);
      }
    }
  }

  /* will compute block_size * block_size iteration times each process */
  // for (i = 0; i < iteration; i++) {
  //   for (k = i * block_size; k < block_size * (i + 1); k++) {
  //     double complex arg = (k * offset / (double)n);
  //     double complex exponent = cexp(-2.0I * M_PI * arg);
  //     for (l = 0; l < block_size; l++) {
  //       local_sum += source.mat[k][l + offset] * exponent;
  //     }
  //     for (l = block_size; l < n - offset; l += block_size) {
  //       exponent *= cexp(-2.0I * M_PI * arg * block_size / (double)n);
  //       for (int m = 0; m < block_size; m++) {
  //         local_sum += source.mat[k][l + m] * exponent;
  //       }
  //     }
  //   }
  // }

  MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

   /* finish time */
  finish = MPI_Wtime();
  loc_elapsed = finish - start;
  MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  
  sum /= source.size;

  if(world_rank == 0){
    printf("Elapsed time: %e seconds\n", elapsed);
    printf("Average : (%lf, %lf)", creal(sum), cimag(sum));
  }
  
  MPI_Finalize();

  return 0;
}