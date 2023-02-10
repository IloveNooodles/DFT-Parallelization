OUTPUT_FOLDER = bin

all: serial parallel

parallel:
# TODO : Parallel compilation

serial:
	gcc src/serial/c/serial.c -o $(OUTPUT_FOLDER)/serial -lm