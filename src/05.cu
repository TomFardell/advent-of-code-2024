#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "utils.cuh"

#define FILE_NAME "data/05p.txt"
#define NUM_RULES 1176   // s: 21, p: 1176
#define NUM_MANUALS 192  // s: 6, p: 192
#define MAX_NUMS_IN_MANUAL 24
#define BLOCK_SIZE 1

#define BUFFER_SIZE (MAX_NUMS_IN_MANUAL * 3 + 2)

__constant__ int rules_dev[NUM_RULES][2];
__constant__ int manuals_dev[NUM_MANUALS][MAX_NUMS_IN_MANUAL];

__device__ int manual_is_valid(const int *manual) {
  for (int r = 0; r < NUM_RULES; r++) {
    int found_second = 0;
    for (int m = 0; manual[m] != -1; m++) {
      if (manual[m] == rules_dev[r][1]) found_second = 1;  // Found the second element of the rule
      if (manual[m] == rules_dev[r][0]) {
        if (found_second) return 0;  // Second element of the rule came before the first
        break;                       // First element of the rule came first
      }
    }
  }

  return 1;
}

// Assumes manuals have odd number of elements (for even would return upper of middle two)
__device__ int get_middle_element(const int *manual) {
  int n;
  for (n = 0; manual[n] != -1; n++);  // Increment n until end of manual found

  return manual[n / 2];
}

__device__ int get_middle_after_reordering(const int *manual) {
  int reorder[MAX_NUMS_IN_MANUAL];
  int i;

  // Bubble sort. Could merge sort on host with comparison respective to rules, but would lose out
  // on parallelism
  for (i = 0; manual[i] != -1; i++) {
    reorder[i] = manual[i];
    reorder[i + 1] = -1;
    for (int j = i; !manual_is_valid(reorder); j--) {
      int temp = reorder[j];
      reorder[j] = reorder[j - 1];
      reorder[j - 1] = temp;
    }
  }

  return reorder[i / 2];
}

// Could use a local sum, but makes the code really messy and already did this a bunch of times
// for earlier puzzles
__global__ void sum_middles(int *total_valid, int *total_reordered) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= NUM_MANUALS) return;

  if (manual_is_valid(manuals_dev[i])) {
    atomicAdd(total_valid, get_middle_element(manuals_dev[i]));
  } else {
    atomicAdd(total_reordered, get_middle_after_reordering(manuals_dev[i]));
  }
}

int main(void) {
  char line_buffer[BUFFER_SIZE];
  int rules[NUM_RULES][2];
  int manuals[NUM_MANUALS][MAX_NUMS_IN_MANUAL];

  FILE *file = fopen(FILE_NAME, "r");
  for (int l = 0; l < NUM_RULES; l++) {
    fgets(line_buffer, BUFFER_SIZE, file);
    sscanf(line_buffer, "%d|%d", &rules[l][0], &rules[l][1]);
  }
  fgets(line_buffer, BUFFER_SIZE, file);  // Read empty line
  for (int l = 0; l < NUM_MANUALS; l++) {
    fgets(line_buffer, BUFFER_SIZE, file);
    int c = 0;
    while (line_buffer[c + 2] == ',') {
      sscanf(line_buffer + c, "%d", &manuals[l][c / 3]);
      c += 3;
    }
    sscanf(line_buffer + c, "%d", &manuals[l][c / 3]);  // No comma after last entry
    manuals[l][(c / 3) + 1] = -1;                       // -1 signifies the end of the manual
  }
  fclose(file);

  error_check(cudaMemcpyToSymbol(rules_dev, rules, NUM_RULES * 2 * sizeof(int)));
  error_check(
      cudaMemcpyToSymbol(manuals_dev, manuals, NUM_MANUALS * MAX_NUMS_IN_MANUAL * sizeof(int)));
  int *valid_sum_dev, *reordered_sum_dev;
  error_check(cudaMalloc(&valid_sum_dev, sizeof(int)));
  error_check(cudaMemset(valid_sum_dev, 0, sizeof(int)));
  error_check(cudaMalloc(&reordered_sum_dev, sizeof(int)));
  error_check(cudaMemset(reordered_sum_dev, 0, sizeof(int)));

  sum_middles<<<calculate_num_blocks(BLOCK_SIZE, NUM_MANUALS), BLOCK_SIZE>>>(valid_sum_dev,
                                                                             reordered_sum_dev);
  error_check(cudaPeekAtLastError());

  int valid_sum, reordered_sum;
  error_check(cudaMemcpy(&valid_sum, valid_sum_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaFree(valid_sum_dev));
  error_check(cudaMemcpy(&reordered_sum, reordered_sum_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaFree(reordered_sum_dev));

  printf("%d\n%d\n", valid_sum, reordered_sum);
}