CUDAC = nvcc
CFLAGS = -Wall -Wextra
CUDAFLAGS = "--compiler-options=$(CFLAGS)" -rdc=true

LD = nvcc

all: \
	01a

01a: src/01a.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

%.o: %.cu
	$(CUDAC) $(CUDAFLAGS) -c -o $@ $<

clean:
	rm -f *.out src/*.o src/*.out