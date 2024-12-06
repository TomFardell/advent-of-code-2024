CUDAC = nvcc
CFLAGS = -Wall -Wextra
CUDAFLAGS = "--compiler-options=$(CFLAGS)" -rdc=true

LD = nvcc
LDFLAGS = -lm

all: \
	01 02 03 04 05

01: src/01.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

02: src/02.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

03: src/03.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

04: src/04.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

05: src/05.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

%.o: %.cu
	$(CUDAC) $(CUDAFLAGS) $(EXTRAFLAGS) -c -o $@ $<

clean:
	rm -f *.out src/*.o src/*.out