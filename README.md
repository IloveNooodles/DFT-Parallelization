# Tugas Kecil - Paralel DFT

Repository untuk kit tugas besar IF3230 Sistem Paralel dan Terdistribusi 2023

| Nim      | Nama                           |
| -------- | ------------------------------ |
| 13520016 | Gagas Praharsa Bahar           |
| 13520029 | Muhammad Garebaldhie ER Rahman |
| 13520036 | I Gede Arya R. P               |
| 13520044 | Adiyansa Prasetya Wicaksana    |

Made with love from ayah_dan_ibu

## MPI

What have been implemented

- DFT Parallel
- FFT Serial
- FFT Parallel

Benchmark
| Test Case | Serial     | Serial FFT | Parallel | Parallel FFT |
| --------- | ---------- | ---------- | -------- | ------------ |
| 32        | 0.039s     | 4.23e-4 s  | 0.0037s  | 2.07e-4 s    |
| 64        | 0.628s     | 2.59e-3 s  | 0.61s    | 1.58e-3s s   |
| 128       | 12.7s      | 1.06e-2 s  | 9s       | 3.55e-3 s    |
| 256       | 3m 54s     | 0.0357s    | 1m 24s   | 3.08e-2 s    |
| 512       | 71m 6.124s | 0.14s      | 35m      | 7.84e-2 s    |

## MP

What have been implemented

- DFT Parallel
- FFT Serial
- FFT Parallel


## Cuda

What have been implemented

- DFT Parallel

Benchmark
| Test Case | Serial     | Parallel |
| --------- | ---------- | -------- |
| 32        | 0.039s     | 0.0030s  |
| 64        | 0.628s     | 0.077s   |
| 128       | 12.7s      | 0.345s   |
| 256       | 3m 54s     | 2.34s    |
| 512       | 71m 6.124s | 32s      |