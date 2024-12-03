#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.cuh"

#define FILE_NAME "data/02p.txt"
#define NUM_LINES 1000
#define MAX_NUMS_ON_LINE 9  // Set to one more than actual line lengths to house array terminator
#define MAX_NUM_DIGITS 2
#define MAX_LINE_LENGTH 30
#define BLOCK_SIZE 4

__constant__ int data_dev[NUM_LINES * MAX_NUMS_ON_LINE];

__device__ int is_safe(int *nums) {
  if (nums[0] == nums[1]) {
    return 0;
  }

  // 1 if increasing, -1 if decreasing
  int sign = 2 * (nums[0] < nums[1]) - 1;

  for (int i = 0; nums[i + 1] != -1; i++) {
    int diff = sign * (nums[i + 1] - nums[i]);
    if (diff < 1 || 3 < diff) {
      return 0;
    }
  }

  return 1;
}

__device__ int is_safe_with_removal(int *nums) {
  // Simply check all lists with one number removed
  for (int i = 0; nums[i] != -1; i++) {
    int nums_aug[MAX_NUMS_ON_LINE];

    int j;
    for (j = 0; nums[j] != -1; j++) {
      // Once j exceeds i, write tto
      nums_aug[j - (j > i)] = nums[j];
    }
    nums_aug[j - 1] = -1;

    if (is_safe(nums_aug)) return 1;
  }

  return is_safe(nums);
}

// Uses data stored in constant memory
__global__ void count_safe(int *count) {
  int l = blockIdx.x * blockDim.x + threadIdx.x;

  if (l >= NUM_LINES) return;

  __shared__ int block_count;
  block_count = 0;
  __syncthreads();

  atomicAdd(&block_count, is_safe(data_dev + (l * MAX_NUMS_ON_LINE)));
  __syncthreads();

  if (threadIdx.x == 0) atomicAdd(count, block_count);
}

__global__ void count_safe_with_removal(int *count) {
  int l = blockIdx.x * blockDim.x + threadIdx.x;

  if (l >= NUM_LINES) return;

  __shared__ int block_count;
  block_count = 0;
  __syncthreads();

  atomicAdd(&block_count, is_safe_with_removal(data_dev + (l * MAX_NUMS_ON_LINE)));
  __syncthreads();

  if (threadIdx.x == 0) atomicAdd(count, block_count);
}

int main(void) {
  char line_buffer[MAX_LINE_LENGTH];
  char num_buffer[MAX_NUM_DIGITS + 1];
  int data[NUM_LINES * MAX_NUMS_ON_LINE];

  FILE *file = fopen(FILE_NAME, "r");

  // Messy file read loop. Puts numbers into 2D array where each row is a line. Value of -1
  // signifies that the line has ended
  for (int l = 0; l < NUM_LINES; l++) {
    fgets(line_buffer, MAX_LINE_LENGTH, file);

    int n = 0, p = 0;
    for (int i = 0; i < MAX_LINE_LENGTH; i++) {
      if (line_buffer[i] == '\n' || line_buffer[i] == ' ' || line_buffer[i] == '\0') {
        int num;
        sscanf(num_buffer, "%d", &num);
        data[l * MAX_NUMS_ON_LINE + n] = num;
        n++;

        // Empty and reset num_buffer
        for (int j = 0; j < MAX_NUM_DIGITS; j++) {
          num_buffer[j] = ' ';
        }
        p = 0;

        // Break if this is the end of the line
        if (line_buffer[i] != ' ') {
          data[l * MAX_NUMS_ON_LINE + n] = -1;
          break;
        }
      } else {
        num_buffer[p] = line_buffer[i];
        p++;
      }
    }
  }

  fclose(file);

  error_check(cudaMemcpyToSymbol(data_dev, data, NUM_LINES * MAX_NUMS_ON_LINE * sizeof(int)));

  int *count_dev;
  error_check(cudaMalloc(&count_dev, sizeof(int)));
  error_check(cudaMemset(count_dev, 0, sizeof(int)));

  int *count_with_removal_dev;
  error_check(cudaMalloc(&count_with_removal_dev, sizeof(int)));
  error_check(cudaMemset(count_with_removal_dev, 0, sizeof(int)));

  int num_blocks = calculate_num_blocks(BLOCK_SIZE, NUM_LINES);
  count_safe<<<num_blocks, BLOCK_SIZE>>>(count_dev);
  count_safe_with_removal<<<num_blocks, BLOCK_SIZE>>>(count_with_removal_dev);

  int count;
  error_check(cudaMemcpy(&count, count_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaFree(count_dev));

  int count_with_removal;
  error_check(
      cudaMemcpy(&count_with_removal, count_with_removal_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaFree(count_with_removal_dev));

  printf("%d\n%d\n", count, count_with_removal);

  return 0;
}