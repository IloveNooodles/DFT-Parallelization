/* compile: mpicc mpi.c -o mpi */
/* run: mpirun -n 4 ./bin/parallel_mpi*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <iostream>
#include <vector>

using namespace std::complex_literals;
using namespace std;

/* Matrix initialization and declarations */
constexpr complex<double> pi()
{
  return std::atan(1) * 4;
}

class Matrix
{
private:
  int n;
  vector<vector<double>> *data_ptr;
  vector<vector<double>> &data;

public:
  Matrix() : data_ptr(new vector<vector<double>>()), data(*data_ptr)
  {
    this->data.resize(10, vector<double>(10, 0));
  }
  ~Matrix()
  {
    delete data_ptr;
  }
  /* Read input only from process 0 can't be parallelized */
  void readMatrix()
  {
    cin >> this->n;
    this->data.resize(this->n, vector<double>(this->n, 0.0));
    for (int i = 0; i < this->n; i++)
      for (int j = 0; j < this->n; j++)
        cin >> this->data[i][j];
  }
  int size()
  {
    return this->n;
  }
  /* TODO: parallelize this */
  complex<double> dftElement(int k, int l)
  {
    complex<double> element = 0;
    for (int m = 0; m < this->n; m++)
    {
      for (int n = 0; n < this->n; n++)
      {
        complex<double> sample = (k * m / (double)this->n) + (l * n / (double)this->n);
        complex<double> exponent = exp(-2.0i * pi() * sample);
        element += this->data[m][n] * exponent;
      }
    }
    return element / (complex<double>)(this->n * this->n);
  }
};

/* util functions */
void read_vector(int world_rank, MPI_Comm comm)
{
  /* use scatter to only give needed data */
  if (world_rank == 0)
  {
    // MPI_Scatter()
  }
  else
  {
    // MPI_Scatter()
  }
}

void recv_vector(int world_rank, MPI_Comm comm)
{
  if (world_rank == 0)
  {
    // MPI_Gather();
  }
  else
  {
    // MPI_Gather();
  }
}

int main(void)
{
  /* global variable */
  Matrix *m = NULL;
  int world_size;
  int world_rank;
  int n;
  double elapsed, sum;

  /* local variable */
  int local_n;
  double local_sum;
  double start, finish, loc_elapsed;

  /* init the MPI process */
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  /* read matrix from stdin and set size iff worldrank 0 */
  if (world_rank == 0)
  {
    m = new Matrix();
    m->readMatrix();
    n = m->size();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* get start time */
  start = MPI_Wtime();

  /* finish time */
  finish = MPI_Wtime();
  loc_elapsed = finish - start;
  MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (world_rank == 0)
  {

    printf("Elapsed time = %.15f seconds \n", elapsed);
  }
  else
  {
    printf("Im coming from: %d\n", world_rank);
  }

  MPI_Finalize();

  return 0;
}