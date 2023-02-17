OUTPUT_FOLDER = bin

all: serial parallel

parallel:
	mpicc src/open-mpi/dft.c -g -Wall -o $(OUTPUT_FOLDER)/parallel_mpi -lm
	gcc src/open-mp/dft.c --openmp -g -Wall -o $(OUTPUT_FOLDER)/parallel_mp -lm

serial:
	gcc src/serial/c/serial.c -o $(OUTPUT_FOLDER)/serial -lm
	g++ src/serial/c++/serial.cpp -o $(OUTPUT_FOLDER)/serial_cpp -lm

clean:
	find ./bin -type f -not -name .gitignore -delete