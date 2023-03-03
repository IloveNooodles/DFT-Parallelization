# Parallelization of DFT with OpenMP

## Implemented Bonus

1. Serial FFT
2. Parallel FFT

## DFT

The OpenMP implementation of the parallel DFT version is not much different from the serial algorithm and the algorithm used in the past OpenMPI implementation.

### Read input

In the OpenMP implementation, the input is read by the master thread and processed as usual.

### Optimize summation

In the OpenMP implementation, the optimization is done by adding #pragma directives. The compiler then takes care of the parallelization of the code.

On our code, we used the following code to optimize the parallelization of the summation:

```c
  #pragma omp parallel shared(source, result) num_threads(8)
  {
    #pragma omp for collapse(2) schedule(static)
    for (k = 0; k < source.size; k++){
      for (l = 0; l < source.size; l++){
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
```

On the first part of the summation, we used the `collapse` directive to collapse the two loops into one. This is done to make the code more readable and to make the compiler optimize the code better. We also used the `schedule(static)` directive to make the compiler schedule the threads use cyclic schedule. This is done to make the compiler schedule the threads in a way that is more efficient and gain more speedup.

Also, we used the `#pragma omp parallel shared(source, result) num_threads(8)` directive to make the compiler use 8 threads. This is done to make the compiler use the maximum number of threads available. This directive is also used to make the compiler know that the `source` and `result` variables are shared between the threads.

On the second part of the summation, we used the `reduction(+: sum)` directive to make the compiler do the summation in parallel. This directive also means that variable `sum` is the reduction variable and combined using the `+` operator across all threads. We also used the `schedule(static)` directive to make the compiler schedule the threads statically. This is done to make the compiler schedule the threads in a way that is more efficient.

## FFT

The OpenMP implementation of the parallel FFT version is not much different from the serial FFT algorithm and the algorithm used in the past OpenMPI implementation.

The main piece of parallelization code is shown here:

```c

   #pragma omp parallel
    {
        #pragma omp single nowait
        {
            fft_2d(mat, rowLen, rowLen*rowLen);
        }
    }

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < rowLen*rowLen; i++) {
        sum += mat[i];
    }
    sum /= rowLen*rowLen*rowLen;

```

The first part of the code is the parallelization of the FFT algorithm. We used the `#pragma omp parallel` directive to make the compiler use all the threads available. We also used the `#pragma omp single nowait` directive to make the compiler schedule the threads in a way that is more efficient. This directive also means that the threads will not wait for each other to finish.

Lastly, we used the `#pragma omp parallel for reduction(+:sum)` directive to make the compiler do the summation in parallel. This directive also means that variable `sum` is the reduction variable and combined using the `+` operator across all threads.
