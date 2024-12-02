CUDAC = nvcc
CFLAGS = -Wall -Wextra
CUDAFLAGS = "--compiler-options=$(CFLAGS)" -rdc=true

LD = nvcc

all: \
	01

01: src/01.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

%.o: %.cu
	$(CUDAC) $(CUDAFLAGS) -c -o $@ $<

clean:
	rm -f *.out src/*.o src/*.out