#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuComplex.h>

#define MAX_N 512

void readMatrix(int *n, cuDoubleComplex *m)
{
  scanf("%d", n);
  for (int i = 0; i < *n; i++){
    for (int j = 0; j < *n; j++){
      scanf("%lf", &m[i * *n + j].x);
    }
  }
};

__global__ void dft2d_kernel(cuDoubleComplex *in, cuDoubleComplex *out, int width, int height, int dir)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    // double sum_real = 0;
    // double sum_imag = 0;

    cuDoubleComplex sum = make_cuDoubleComplex(0.0f, 0.0f);

    for (int k = 0; k < height; k++)
    {
        for (int l = 0; l < width; l++)
        {
            double angle = 2 * M_PI * (dir == 1 ? (x * k / (double)width + y * l / (double)height) : (x * k / (double)width - y * l / (double)height));
            
            cuDoubleComplex twiddle = make_cuDoubleComplex(cos(angle), sin(angle));
            cuDoubleComplex temp = cuCmul(in[k * width + l], twiddle);

            sum = cuCadd(sum, temp);
            
            // sum_real += in[k * width + l].x * cos(angle);
            // sum_imag += -in[k * width + l].x * sin(angle);
        }
    }


    out[idx] = cuCmul(make_cuDoubleComplex((dir == 1 ? 1.0 : 1.0 / (width * height)), 0.0), sum);
    // out[idx].x = (dir == 1 ? 1 : 1.0 / (width * height)) * sum_real;
    // out[idx].y = (dir == 1 ? 1 : 1.0 / (width * height)) * sum_imag;
}

void dft2d_cuda(cuDoubleComplex *d_idata, cuDoubleComplex *d_odata, int width, int height, int dir)
{
    cuDoubleComplex *d_in, *d_out;
    cudaMalloc(&d_in, width * height * sizeof(cuDoubleComplex));
    cudaMalloc(&d_out, width * height * sizeof(cuDoubleComplex));

    cudaMemcpy(d_in, d_idata, width * height * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size(width / block_size.x, height / block_size.y);
    dft2d_kernel<<<grid_size, block_size>>>(d_in, d_out, width, height, dir);

    cudaMemcpy(d_odata, d_out, width * height * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main(void) 
{
  cuDoubleComplex *source = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * MAX_N * MAX_N);
  cuDoubleComplex *result = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * MAX_N * MAX_N);

  int n, k, l;
  cuDoubleComplex sum = make_cuDoubleComplex(0.0f, 0.0f);

  readMatrix(&n, source);


  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);


  cudaEventRecord(start_event, 0);

  // Dir: 1 for forward, -1 for inverse
  dft2d_cuda(source, result, n, n, -1);


  for (k = 0; k < n; k++){
    for (l = 0; l < n; l++){
      sum.x += result[k * n + l].x;
      sum.y += result[k * n + l].y;
    }
  }

  cudaEventRecord(stop_event, 0);

  float elapsed_time = 0.0f;
  cudaEventElapsedTime(&elapsed_time, start_event, stop_event);


  sum.x /= n;
  sum.y /= n;
  printf("Elapsed time: %lf milliseconds\n", elapsed_time);
  printf("Average : (%lf, %lf)", sum.x, sum.y);
}
