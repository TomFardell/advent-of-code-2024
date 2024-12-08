#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "utils.cuh"

#define INPUT 'p'

#if INPUT == 'p'
#define FILE_NAME "data/07p.txt"
#define N 850
#elif INPUT == 's'
#define FILE_NAME "data/07s.txt"
#define N 9
#endif

#define L 15  // Max nums on line
#define BUFFER_SIZE 50
#define BLOCK_SIZE 16

typedef unsigned long long u64;  // CUDA bloats my code enough already!

__constant__ u64 totals_dev[N];
__constant__ int nums_dev[N * L];
__constant__ int line_lengths_dev[N];

__device__ int can_make(const u64 total, const int *nums, const int n) {
  if (n == 1) return (total <= UINT32_MAX && (int)total == nums[0]);  // Check cast overflow first
  if (total <= 0) return 0;

  int mul = 0;
  if (total % nums[n - 1] == 0) mul = can_make(total / nums[n - 1], nums, n - 1);
  int add = can_make(total - nums[n - 1], nums, n - 1);

  return (add || mul);
}

// See if larger number has the smaller number as its suffix
__device__ int num_has_suffix(const u64 n_large, const int n_small) {
  u64 nl = n_large;
  for (int ns = n_small; ns > 0; nl /= 10, ns /= 10) {
    if ((int)(nl % 10) != ns % 10) {
      return 0;
    }
  }
  return 1;
}

__device__ int can_make_with_concat(const u64 total, const int *nums, const int n) {
  if (n == 1) return (total <= UINT32_MAX && (int)total == nums[0]);  // Check cast overflow first
  if (total <= 0) return 0;

  int mul = 0;
  if (total % nums[n - 1] == 0) mul = can_make_with_concat(total / nums[n - 1], nums, n - 1);

  int concat = 0;
  if (num_has_suffix(total, nums[n - 1])) {
    u64 new_total = total;
    for (int ns = nums[n - 1]; ns > 0; ns /= 10) {
      new_total /= 10;
    }
    concat = can_make_with_concat(new_total, nums, n - 1);
  }

  int add = can_make_with_concat(total - nums[n - 1], nums, n - 1);

  return (add || mul || concat);
}

__global__ void sum_calibrated(u64 *total) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= N) return;

  atomicAdd(total, totals_dev[i] * can_make(totals_dev[i], nums_dev + L * i, line_lengths_dev[i]));
}

__global__ void sum_calibrated_with_concat(u64 *total) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= N) return;

  atomicAdd(total, totals_dev[i] *
                       can_make_with_concat(totals_dev[i], nums_dev + L * i, line_lengths_dev[i]));
}

int main(void) {
  char line_buffer[BUFFER_SIZE];
  u64 totals[N];
  int nums[N * L];
  int line_lengths[N];

  FILE *file = fopen(FILE_NAME, "r");
  for (int l = 0; l < N; l++) {
    fgets(line_buffer, BUFFER_SIZE, file);
    sscanf(line_buffer, "%llu", totals + l);

    int c = 0;
    // Iterate through numbers after the colon and first space
    for (char *p = strchr(line_buffer, ' '); p != NULL; p = strchr(p + 1, ' '), c++) {
      sscanf(p, "%d", nums + (c + L * l));
    }
    line_lengths[l] = c;
  }
  fclose(file);

  error_check(cudaMemcpyToSymbol(totals_dev, totals, N * sizeof(u64)));
  error_check(cudaMemcpyToSymbol(nums_dev, nums, N * L * sizeof(int)));
  error_check(cudaMemcpyToSymbol(line_lengths_dev, line_lengths, N * sizeof(int)));

  u64 *total_dev;
  error_check(cudaMalloc(&total_dev, sizeof(u64)));
  error_check(cudaMemset(total_dev, 0, sizeof(u64)));

  u64 *total_with_concat_dev;
  error_check(cudaMalloc(&total_with_concat_dev, sizeof(u64)));
  error_check(cudaMemset(total_with_concat_dev, 0, sizeof(u64)));

  sum_calibrated<<<calculate_num_blocks(BLOCK_SIZE, N), BLOCK_SIZE>>>(total_dev);
  error_check(cudaPeekAtLastError());
  sum_calibrated_with_concat<<<calculate_num_blocks(BLOCK_SIZE, N), BLOCK_SIZE>>>(
      total_with_concat_dev);
  error_check(cudaPeekAtLastError());

  u64 total;
  error_check(cudaMemcpy(&total, total_dev, sizeof(u64), cudaMemcpyDeviceToHost));
  error_check(cudaFree(total_dev));
  u64 total_with_concat;
  error_check(
      cudaMemcpy(&total_with_concat, total_with_concat_dev, sizeof(u64), cudaMemcpyDeviceToHost));
  error_check(cudaFree(total_with_concat_dev));

  printf("%llu\n%llu\n", total, total_with_concat);

  return 0;
}