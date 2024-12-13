#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"

#define INPUT 'p'

#if INPUT == 'p'
#define FILE_NAME "data/10p.txt"
#define N 55
#elif INPUT == 's'
#define FILE_NAME "data/10s.txt"
#define N 8
#endif

#define BLOCK_WIDTH 4

typedef struct {
  int i1, i2;
} int_pair;

__constant__ int directions_dev[8] = {0, 1, 0, -1, 1, 0, -1, 0};
__constant__ int map_dev[N * N];

__device__ int in_grid(int i, int j) { return (0 <= i && i < N && 0 <= j && j < N); }

__device__ int_pair count_paths(int i, int j) {
  int_pair counts = {0};       // Counts for part 1 and 2, since using DFS for both
  int point_stack[3 * N * N];  // (i, j, height)
  int seen_nines[N * N] = {0};
  point_stack[0] = i;
  point_stack[1] = j;
  point_stack[2] = 0;

  int sp = 0;
  while (sp >= 0) {
    int si = point_stack[3 * sp];
    int sj = point_stack[3 * sp + 1];
    int sh = point_stack[3 * sp + 2];
    sp--;  // Decrement the stack pointer, effectively popping this element

    for (int d = 0; d < 4; d++) {
      int ni = directions_dev[2 * d] + si;
      int nj = directions_dev[2 * d + 1] + sj;
      // If adjacent square has next level up, add it to the stack
      if (in_grid(ni, nj) && map_dev[ni + N * nj] == sh + 1) {
        if ((sh + 1) == 9) {
          if (!seen_nines[ni + N * nj]) {
            seen_nines[ni + N * nj] = 1;
            counts.i1++;
          }
          counts.i2++;
        } else {
          sp++;
          point_stack[3 * sp] = ni;
          point_stack[3 * sp + 1] = nj;
          point_stack[3 * sp + 2] = sh + 1;
        }
      }
    }
  }

  return counts;
}

__global__ void sum_paths(int_pair *totals) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (i >= N || j >= N) return;
  if (map_dev[i + N * j] != 0) return;  // Only proceed from valid start points

  int_pair counts = count_paths(i, j);
  atomicAdd(&totals->i1, counts.i1);
  atomicAdd(&totals->i2, counts.i2);
}

int main(void) {
  int map[N * N];

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      map[i + N * j] = fgetc(file) - '0';
    }
    fgetc(file);  // Read newline
  }
  fclose(file);

  error_check(cudaMemcpyToSymbol(map_dev, map, N * N * sizeof(int)));

  int_pair *totals_dev;
  error_check(cudaMalloc(&totals_dev, sizeof(int_pair)));
  error_check(cudaMemset(totals_dev, 0, sizeof(int_pair)));

  int block_count_width = calculate_num_blocks(BLOCK_WIDTH, N);
  sum_paths<<<dim3(block_count_width, block_count_width), dim3(BLOCK_WIDTH, BLOCK_WIDTH)>>>(
      totals_dev);
  error_check(cudaPeekAtLastError());

  int_pair totals;
  error_check(cudaMemcpy(&totals, totals_dev, sizeof(int_pair), cudaMemcpyDeviceToHost));
  error_check(cudaFree(totals_dev));

  printf("%d\n%d\n", totals.i1, totals.i2);

  return 0;
}
