# Parallelization of DFT algorithm using CUDA

This code provides a CUDA implementation of the 2D Discrete Fourier Transform (DFT) algorithm. The DFT is a mathematical transform that transforms a 2D input signal into its frequency domain representation. It has many applications in signal processing, image processing, and data compression.

## DFT
The code reads a square matrix from standard input and performs a 2D DFT on it. The main function reads the input matrix using the `readMatrix` function and then calls the `dft2d_cuda` function to perform the 2D DFT. The resulting frequency-domain representation is stored in the result array.

The `dft2d_cuda function` performs the 2D DFT on the input matrix using CUDA. It first allocates device memory for the input and output matrices using `cudaMalloc`. It then copies the input matrix from host memory to device memory using `cudaMemcpy`. The code is below:
```c
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
```

The `dft2d_kernel` function is the CUDA kernel that performs the actual DFT computation. Each thread in the CUDA grid computes one element of the output matrix using the input matrix and the DFT formula. The output matrix is written back to device memory. The code is below:
```c
__global__ void dft2d_kernel(cuDoubleComplex *in, cuDoubleComplex *out, int width, int height, int dir)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    cuDoubleComplex sum = make_cuDoubleComplex(0.0f, 0.0f);

    for (int k = 0; k < height; k++)
    {
        for (int l = 0; l < width; l++)
        {
            double angle = 2 * M_PI * (dir == 1 ? (x * k / (double)width + y * l / (double)height) : (x * k / (double)width - y * l / (double)height));
            
            cuDoubleComplex twiddle = make_cuDoubleComplex(cos(angle), sin(angle));
            cuDoubleComplex temp = cuCmul(in[k * width + l], twiddle);

            sum = cuCadd(sum, temp);
        }
    }

    out[idx] = cuCmul(make_cuDoubleComplex((dir == 1 ? 1.0 : 1.0 / (width * height)), 0.0), sum);
}

```

After the CUDA kernel finishes executing, the output matrix is copied back from device memory to host memory using cudaMemcpy. The host then calculates the average of the output matrix and prints it to standard output along with the elapsed time taken by the DFT computation.

## CUDA Implementation
The CUDA implementation parallelizes the DFT computation across multiple threads in a CUDA grid. Each thread in the grid computes one element of the output matrix using the input matrix and the DFT formula. The CUDA kernel uses two-dimensional thread blocks and grid dimensions to efficiently compute the DFT.

The CUDA implementation uses the `cuComplex` library, which provides support for complex numbers in CUDA. The `cuDoubleComplex` data type is used to represent complex numbers with double precision.

The DFT algorithm is computationally intensive and can benefit from parallelization. The CUDA implementation provides significant speedup over the serial implementation.

## Usage

To run the code, simply compile it using a CUDA compiler and run the resulting executable. The code reads the input matrix from standard input, so you can provide the input matrix in a text file and redirect it to standard input when running the executable.

```
nvcc cuda.cu -o cuda
./cuda < input.txt
```

**Note that you must have a CUDA-enabled GPU and the CUDA toolkit installed to run this code.**