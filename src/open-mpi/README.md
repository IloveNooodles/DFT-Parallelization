# Parallelization of DFT algorithm using Open MPI

## Implemented Bonus

1. Serial FFT
2. Parallel FFT

## DFT

The parallel dft version is basically the same as serial one but we added optimize on the `sum` part of the algorithm iself. Basically we can sum of what we did into two part

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

we divide the matrix n x n depends of the number of process that being runned by dividing the size of matrix by the `world size`. And then

```c
int block_size = n / world_size;
```

And then if we had 32 x 32 matrix, and after dividing it to the block size we got `16`. So each process will calculate atleast `16 x 16` matrix. But if we're only using 2 cores than the summation isn't complete so we use iteration for each process to count how many times each process need to calculate

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

1. each of process will calculate `block_size x block_size` matrix for `iteration` times.
2. After the local sum is being calculated, we use `MPI_Reduce` to sent the sum to the `sum` global variable. and then we put `MPI_Finalize` in the end to end the process to state that MPI Process is done

## FFT
