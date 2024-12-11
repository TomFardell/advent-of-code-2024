CUDAC = nvcc
CFLAGS = -Wall -Wextra
CUDAFLAGS = "--compiler-options=$(CFLAGS)" -rdc=true

LD = nvcc
LDFLAGS = -lm

all: \
	01 02 03 04 05 06 07

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

06: src/06.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

07: src/07.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

08: src/08.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

09: src/09.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

10: src/10.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

%.o: %.cu
	$(CUDAC) $(CUDAFLAGS) $(EXTRAFLAGS) -c -o $@ $<

clean:
	rm -f *.out src/*.o src/*.out
