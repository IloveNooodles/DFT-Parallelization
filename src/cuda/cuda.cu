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

#define MAX_N 512

void readMatrix(int *n, float2 *m)
{
  scanf("%d", n);
  for (int i = 0; i < *n; i++){
    for (int j = 0; j < *n; j++){
      scanf("%f", &m[i * *n + j].x);
    }
  }
};

__global__ void dft2d_kernel(float2 *in, float2 *out, int width, int height, int dir)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    float sum_real = 0;
    float sum_imag = 0;

    for (int k = 0; k < height; k++)
    {
        for (int l = 0; l < width; l++)
        {
            float angle = 2 * M_PI * (dir == 1 ? (x * k / (float)width + y * l / (float)height) : (x * k / (float)width - y * l / (float)height));
            sum_real += in[k * width + l].x * cos(angle);
            sum_imag += -in[k * width + l].x * sin(angle);
        }
    }

    out[idx].x = (dir == 1 ? 1 : 1.0 / (width * height)) * sum_real;
    out[idx].y = (dir == 1 ? 1 : 1.0 / (width * height)) * sum_imag;
}

void dft2d_cuda(float2 *d_idata, float2 *d_odata, int width, int height, int dir)
{
    float2 *d_in, *d_out;
    cudaMalloc(&d_in, width * height * sizeof(float2));
    cudaMalloc(&d_out, width * height * sizeof(float2));

    cudaMemcpy(d_in, d_idata, width * height * sizeof(float2), cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size(width / block_size.x, height / block_size.y);
    dft2d_kernel<<<grid_size, block_size>>>(d_in, d_out, width, height, dir);

    cudaMemcpy(d_odata, d_out, width * height * sizeof(float2), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main(void) 
{
  float2 *source = (float2*)malloc(sizeof(float2) * MAX_N * MAX_N);
  float2 *result = (float2*)malloc(sizeof(float2) * MAX_N * MAX_N);

  int n, k, l;
  float2 sum = make_float2(0.0f, 0.0f);

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
