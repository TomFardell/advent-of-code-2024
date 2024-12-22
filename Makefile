CC = gcc
CUDAC = nvcc
CFLAGS = -Wall -Wextra
CUDAFLAGS = "--compiler-options=$(CFLAGS)" -rdc=true

CLD = gcc
LD = nvcc
LDC = $(LD)
LDFLAGS = -lm
LDCFLAGS = $(LDFLAGS)

all: \
	01 02 03 04 05 06 07 08 09 10 \
	11 12 13 14 15 16 17 18 19 20 \

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

11: src/11.o
	$(LD) -o $@.out $^ $(LDFLAGS)

12: src/12.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

13: src/13.o src/utils.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

14: src/14.o 
	$(LD) -o $@.out $^ $(LDFLAGS)

15: src/15.o
	$(LD) -o $@.out $^ $(LDFLAGS)

16: src/16.o
	$(LD) -o $@.out $^ $(LDFLAGS)

17: src/17.o
	$(LD) -o $@.out $^ $(LDFLAGS)

18: src/18.o
	$(LD) -o $@.out $^ $(LDFLAGS)

19: src/19.o
	$(LD) -o $@.out $^ $(LDFLAGS)

20: src/20.o
	$(LDC) -o $@.out $^ $(LDCFLAGS)

%.o: %.cu
	$(CUDAC) $(CUDAFLAGS) $(EXTRAFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) $(EXTRAFLAGS) -c -o $@ $<

clean:
	rm -f *.out src/*.o src/*.out
