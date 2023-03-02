// gcc mp.c --openmp -o mp
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

#define MAX_N 512

struct Matrix
{
  int size;
  double mat[MAX_N][MAX_N];
};

struct FreqMatrix {
    int    size;
    double complex mat[MAX_N][MAX_N];
};

int readMatrix(int *n, struct Matrix *m)
{
  scanf("%d", &(m->size));
  for (int i = 0; i < m->size; i++){
    for (int j = 0; j < m->size; j++){
      scanf("%lf", &(m->mat[i][j]));
    }
  }

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
  struct Matrix source;
  struct FreqMatrix result;
  
  int n, k, l;
  double complex sum = 0;

  double start, finish;

  n = readMatrix(&n, &source);

  start = omp_get_wtime();

  #pragma omp parallel shared(source, result) num_threads(8) 
  {
    // #pragma omp for collapse(2) reduction(+: sum) schedule(static) nowait
    #pragma omp for collapse(2) schedule(static)
    for (k = 0; k < source.size; k++){
      for (l = 0; l < source.size; l++){
        // sum += dft(&source, k, l);
        result.mat[k][l] = dft(&source, k, l);
      }
    }
  }

  #pragma omp parallel shared(result) num_threads(8) 
  {
    #pragma omp for collapse(2) reduction(+: sum) schedule(static)
    for (k = 0; k < source.size; k++){
      for (l = 0; l < source.size; l++){
        sum += result.mat[k][l];
      }
    }
  }

  finish = omp_get_wtime();

  sum /= source.size;
  printf("Elapsed time: %lf seconds\n", finish-start);
  printf("Average : (%lf, %lf)", creal(sum), cimag(sum));
}