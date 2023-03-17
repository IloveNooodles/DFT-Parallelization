OUTPUT_FOLDER = bin

all: 
	serial parallel

mpi:
	mpicc src/open-mpi/dft.c -g -Wall -o $(OUTPUT_FOLDER)/parallel_dft_mpi -lm
	mpicc src/open-mpi/fft.c -g -Wall -o $(OUTPUT_FOLDER)/parallel_fft_mpi -lm
	gcc src/open-mpi/serial_fft.c -g -Wall -o $(OUTPUT_FOLDER)/serial_fft_mpi -lm

mp:
	gcc src/open-mp/dft.c --openmp -g -Wall -o $(OUTPUT_FOLDER)/parallel_dft_mp -lm
	gcc src/open-mp/fft.c --openmp -g -Wall -o $(OUTPUT_FOLDER)/parallel_fft_mp -lm
	gcc src/open-mp/fft.c --openmp -g -Wall -o $(OUTPUT_FOLDER)/serial_fft_mp -lm

parallel: 
	mpi mp

serial:
	gcc src/serial/c/serial.c -o $(OUTPUT_FOLDER)/serial_dft_c -lm
	g++ src/serial/c++/serial.cpp -o $(OUTPUT_FOLDER)/serial_dft_cpp -lm

clean:
	find ./bin -type f -not -name .gitignore -delete