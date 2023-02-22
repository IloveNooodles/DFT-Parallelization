# Parallelization of DFT algorithm using Open MPI

## Implemented Bonus

1. Serial FFT
2. Parallel FFT

## DFT

The parallel dft version is basically the same as serial one but we added optimize on the `sum` part of the algorithm itself. Basically we can sum of what we did into two part

1. Read input
2. Optimize Summation

### Read input

Beacause of the process with rank `0` only that can accept stdin so we create function readInput to ensure that the process with rank `0` that read the stdin. After the process read, it will broadcast the matrix into all of the avilable process using MPI_Bcast

```c
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
```

### Optimize Summation

In this part we optimize summation for the summation in the main function

```c
double complex sum = 0.0;
    for (int k = 0; k < source.size; k++) {
        for (int l = 0; l < source.size; l++) {
            double complex el = freq_domain.mat[k][l];
            sum += el;
        }
    }
```

We then divide the matrix n x n depending on the number of processes that being ran by dividing the size of matrix by the `world size`. And then

```c
int block_size = n / world_size;
```

If we had 32 x 32 matrix and divided it to the block size we got `16`. So each process will calculate atleast `16 x 16` matrix. But if we're only using 2 cores than the summation isn't complete so we use iteration for each process to count how many times each process need to calculate

```c
int iteration = n * n / (block_size * block_size * world_size);
```

And last part is to create the offset of the loop to ensure each process calculate all of them.

```c
offset = world_rank * block_size;
```

The summation part is written below

```c
local_sum = 0.0;
for(i = 0; i < iteration; i++){
 for(k = i * block_size; k < block_size * (i + 1); k++){
   for(l = 0; l < block_size; l++){
     local_sum += dft(&source, k, l + offset);
   }
 }
}
MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
```

1. Each of process will calculate `block_size x block_size` matrix for `iteration` times.
2. After the local sum is being calculated, we use `MPI_Reduce` to sent the sum to the `sum` global variable. and then we put `MPI_Finalize` in the end to end the process to state that MPI Process is done

## FFT

FFT (Fast Fourier Transform) is another algorithm that can be used to calculate the DFT. The algorithm is basically the same as DFT but the calculation is different. This algorithm allows for higher speeds than DFT.

The algorithm starts off the same with DFT, reading the matrix but also converting it into the complex space.

The main difference is that FFT uses a Divide and Conquer approach. The main FFT code can be seen here:

```c
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
```

The above code shows how FFT works on a one-dimensional matrix. The algorithm could then be extended to 2D matrices (and also be parallelized) using this piece of code:

```c
void fft_2d(cplx mat[], int rowLen, int world_rank, int world_size) {
    int i;
    int block_size = rowLen / world_size;
    int offset = world_rank * block_size;

  for(i = rowLen * offset; i < rowLen * (offset + block_size); i += rowLen) fft(mat+i, rowLen);
    MPI_Gather(mat + rowLen * offset, rowLen * block_size, MPI_DOUBLE_COMPLEX, mat, rowLen * block_size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    transpose_matrix(mat, rowLen);
    MPI_Bcast(mat, rowLen * rowLen, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    for(i = rowLen * offset; i < rowLen * (offset + block_size); i += rowLen) fft(mat+i, rowLen);
    MPI_Gather(mat + rowLen * offset, rowLen * block_size, MPI_DOUBLE_COMPLEX, mat, rowLen * block_size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    transpose_matrix(mat, rowLen);
    MPI_Bcast(mat, rowLen * rowLen, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}
```

The above code parallelizes the FFT while extending it to 2-dimensional matrices. The algorithm essentially repeats the calculation for one dimensional FFT for the rows of the matrix, and then transposes the matrix to calculate for the columns.

The algorithm is parallelized by dividing the calculation into a few block sizes which will be run parallel. After doing the calculation for the rows, the matrix is transposed and the result broadcasted to all processes. The calculation then continues with the calculation of the columns, and ends with a broadcast again.

FFT allows for higher speeds in calculating the DFT of a matrix. One such case leads to a more than 8x improvement, checking out the bonus.
