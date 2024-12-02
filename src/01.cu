#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.cuh"

#define FILE_NAME "data/01p.txt"
#define NUM_LINES 1000
#define MAX_LINE_LENGTH 20
#define BLOCK_SIZE 4 /* BLOCK_SIZE <= NUM_LINES */

__constant__ int l1_dev[NUM_LINES], l2_dev[NUM_LINES];

__device__ int get_score(int num, int *other_list) {
  int l, h;
  int first = 0, last = 0;

  /* Binary search to find first occurrence */
  l = 0, h = NUM_LINES - 1;
  while (l <= h) {
    int m = (l + h) / 2;
    if (num <= other_list[m]) {
      h = m - 1;
    } else {
      l = m + 1;
    }
  }
  first = l;

  /* Return 0 if num not in other_list */
  if (other_list[first] != num) {
    return 0;
  }

  /* Binary search to find (the index after) the last occurrence */
  l = 0, h = NUM_LINES - 1;
  while (l <= h) {
    int m = (l + h) / 2;
    if (num < other_list[m]) {
      h = m - 1;
    } else {
      l = m + 1;
    }
  }
  last = l;

  return (last - first) * num;
}

__global__ void sum_differences(int *total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NUM_LINES) return;

  __shared__ int block_sum;
  block_sum = 0;

  /* Need block_sum initialised before adding to it */
  __syncthreads();

  atomicAdd(&block_sum, abs(l1_dev[i] - l2_dev[i]));

  /* Need all threads to add to block_sum before adding to total */
  __syncthreads();

  /* Only add on the first thread of each block */
  if (threadIdx.x == 0) atomicAdd(total, block_sum);
}

__global__ void sum_scores(int *total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NUM_LINES) return;

  __shared__ int block_sum;
  block_sum = 0;

  /* Need block_sum initialised before adding to it */
  __syncthreads();

  atomicAdd(&block_sum, get_score(l1_dev[i], l2_dev));

  /* Need all threads to add to block_sum before adding to total */
  __syncthreads();

  /* Only add on the first thread of each block */
  if (threadIdx.x == 0) atomicAdd(total, block_sum);
}

int main(void) {
  char buffer[MAX_LINE_LENGTH];
  int l1[NUM_LINES], l2[NUM_LINES];

  FILE *file = fopen(FILE_NAME, "r");

  for (int l = 0; l < NUM_LINES; l++) {
    fgets(buffer, MAX_LINE_LENGTH, file);
    sscanf(buffer, "%d   %d", l1 + l, l2 + l);
  }

  fclose(file);

  int l1_sorted[NUM_LINES], l2_sorted[NUM_LINES];
  merge_sort(l1, NUM_LINES, l1_sorted);
  merge_sort(l2, NUM_LINES, l2_sorted);

  int *total_dev;
  error_check(cudaMalloc(&total_dev, sizeof(int)));
  error_check(cudaMemset(total_dev, 0, sizeof(int)));

  int *score_sum_dev;
  error_check(cudaMalloc(&score_sum_dev, sizeof(int)));
  error_check(cudaMemset(score_sum_dev, 0, sizeof(int)));

  error_check(cudaMemcpyToSymbol(l1_dev, l1_sorted, NUM_LINES * sizeof(int)));
  error_check(cudaMemcpyToSymbol(l2_dev, l2_sorted, NUM_LINES * sizeof(int)));

  sum_differences<<<NUM_LINES / BLOCK_SIZE + 1, BLOCK_SIZE>>>(total_dev);
  sum_scores<<<NUM_LINES / BLOCK_SIZE + 1, BLOCK_SIZE>>>(score_sum_dev);

  int total;
  error_check(cudaMemcpy(&total, total_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaFree(total_dev));

  int score_sum;
  error_check(cudaMemcpy(&score_sum, score_sum_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaFree(score_sum_dev));

  printf("%d\n%d\n", total, score_sum);

  return 0;
}