#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"

#define INPUT 'p'

#if INPUT == 'p'
#define FILE_NAME "data/13p.txt"
#define N 320
#elif INPUT == 's'
#define FILE_NAME "data/13s.txt"
#define N 4
#endif

#define BLOCK_SIZE 8

typedef long long i64;
typedef unsigned long long u64;

typedef struct {
  int ax, ay, bx, by, px, py;
} machine;

__constant__ machine machines_dev[N];

__global__ void calculate_costs(u64 *total1, u64 *total2) {
  int n = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (n >= N) return;

  int ax = machines_dev[n].ax;
  int ay = machines_dev[n].ay;
  int bx = machines_dev[n].bx;
  int by = machines_dev[n].by;
  int px = machines_dev[n].px;
  int py = machines_dev[n].py;
  u64 px2 = px + 10000000000000;
  u64 py2 = py + 10000000000000;

  i64 det = ax * by - bx * ay;
  i64 na = by * px - bx * py;
  i64 nb = ax * py - ay * px;
  i64 det2 = ax * by - bx * ay;
  i64 na2 = by * px2 - bx * py2;
  i64 nb2 = ax * py2 - ay * px2;

  if (det == 0) printf("Zero determiants\n");  // Fortunately, this doesn't print!

  if (det != 0 && (na % det == 0) && (nb % det == 0)) {
    i64 a_presses = na / det;
    i64 b_presses = nb / det;
    u64 cost = 3 * a_presses + b_presses;
    atomicAdd(total1, cost);
  }

  if (det2 != 0 && (na2 % det2 == 0) && (nb2 % det2 == 0)) {
    i64 a_presses = na2 / det2;
    i64 b_presses = nb2 / det2;
    u64 cost = 3 * a_presses + b_presses;
    atomicAdd(total2, cost);
  }
}

int main(void) {
  machine machines[N];

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < N; i++) {
    fscanf(file, "Button A: X+%d, Y+%d\n", &machines[i].ax, &machines[i].ay);
    fscanf(file, "Button B: X+%d, Y+%d\n", &machines[i].bx, &machines[i].by);
    fscanf(file, "Prize: X=%d, Y=%d\n\n", &machines[i].px, &machines[i].py);
  }
  fclose(file);

  error_check(cudaMemcpyToSymbol(machines_dev, machines, N * sizeof(machine)));

  u64 *total1_dev;
  error_check(cudaMalloc(&total1_dev, sizeof(u64)));
  error_check(cudaMemset(total1_dev, 0, sizeof(u64)));
  u64 *total2_dev;
  error_check(cudaMalloc(&total2_dev, sizeof(u64)));
  error_check(cudaMemset(total2_dev, 0, sizeof(u64)));

  calculate_costs<<<calculate_num_blocks(BLOCK_SIZE, N), N>>>(total1_dev, total2_dev);
  error_check(cudaDeviceSynchronize());

  u64 total1;
  error_check(cudaMemcpy(&total1, total1_dev, sizeof(u64), cudaMemcpyDeviceToHost));
  error_check(cudaFree(total1_dev));

  u64 total2;
  error_check(cudaMemcpy(&total2, total2_dev, sizeof(u64), cudaMemcpyDeviceToHost));
  error_check(cudaFree(total2_dev));

  printf("%llu\n%llu\n", total1, total2);

  return 0;
}
