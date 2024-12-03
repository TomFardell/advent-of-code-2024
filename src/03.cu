#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <cuda.h>
#include "utils.cuh"

#define FILE_NAME "data/03p.txt"
#define NUM_LINES 6
#define MAX_LINE_LENGTH 4000
#define MAX_OPERATIONS 800
#define BLOCK_SIZE 16

__constant__ int nums_dev[MAX_OPERATIONS][2];

__global__ void multiply_nums(int n, int *total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  __shared__ int block_total;
  block_total = 0;
  __syncthreads();

  atomicAdd(&block_total, nums_dev[i][0] * nums_dev[i][1]);
  __syncthreads();

  if (threadIdx.x == 0) atomicAdd(total, block_total);
}

int main(void) {
  char line_buffer[MAX_LINE_LENGTH];
  int nums_p1[MAX_OPERATIONS][2] = {0};
  int nums_p2[MAX_OPERATIONS][2] = {0};
  int n1 = 0, n2 = 0;
  int is_on = 1;

  regex_t re_mul;
  if (regcomp(&re_mul, "mul([[:digit:]]\\{1,\\},[[:digit:]]\\{1,\\})", 0)) {
    fprintf(stderr, "REGEX failed to compile");
    exit(EXIT_FAILURE);
  }

  regex_t re_do;
  if (regcomp(&re_do, "do()", 0)) {
    fprintf(stderr, "REGEX failed to compile");
    exit(EXIT_FAILURE);
  }

  regex_t re_dont;
  if (regcomp(&re_dont, "don't()", 0)) {
    fprintf(stderr, "REGEX failed to compile");
    exit(EXIT_FAILURE);
  }

  FILE *file = fopen(FILE_NAME, "r");
  for (int l = 0; l < NUM_LINES; l++) {
    fgets(line_buffer, MAX_LINE_LENGTH, file);

    regmatch_t mul_match;
    // While matches available, find the next match. After execution of the loop contents, move the
    // pointer after the end of the match
    for (char *p = line_buffer; !regexec(&re_mul, p, 1, &mul_match, 0); p += mul_match.rm_eo) {
      int final_do = 0, final_dont = 0;

      regmatch_t do_match;
      // Search for matches of re_do until reaching the start of the next mul match
      for (char *q = p; !regexec(&re_do, q, 1, &do_match, 0) && (q < p + mul_match.rm_so);
           q += do_match.rm_eo) {
        if (do_match.rm_eo < mul_match.rm_eo) final_do += do_match.rm_eo;
      }

      regmatch_t dont_match;
      for (char *q = p; !regexec(&re_dont, q, 1, &dont_match, 0) && (q < p + mul_match.rm_so);
           q += dont_match.rm_eo) {
        if (dont_match.rm_eo < mul_match.rm_eo) final_dont += dont_match.rm_eo;
      }

      if (final_do || final_dont) {
        is_on = (final_do > final_dont);
      }

      int m1, m2;
      sscanf(p + mul_match.rm_so, "mul(%d,%d)", &m1, &m2);

      nums_p1[n1][0] = m1;
      nums_p1[n1][1] = m2;
      n1++;

      if (is_on) {
        nums_p2[n2][0] = m1;
        nums_p2[n2][1] = m2;
        n2++;
      }
    }
  }
  fclose(file);

  regfree(&re_mul);
  regfree(&re_do);
  regfree(&re_dont);

  int *total_p1_dev, *total_p2_dev;
  error_check(cudaMalloc(&total_p1_dev, sizeof(int)));
  error_check(cudaMemset(total_p1_dev, 0, sizeof(int)));
  error_check(cudaMalloc(&total_p2_dev, sizeof(int)));
  error_check(cudaMemset(total_p2_dev, 0, sizeof(int)));

  // Passing n1 and n2 to the kernel, so copying only n1 rows here would work
  error_check(cudaMemcpyToSymbol(*nums_dev, *nums_p1, MAX_OPERATIONS * 2 * sizeof(int)));
  multiply_nums<<<calculate_num_blocks(BLOCK_SIZE, n1), BLOCK_SIZE>>>(n1, total_p1_dev);

  error_check(cudaMemcpyToSymbol(*nums_dev, *nums_p2, MAX_OPERATIONS * 2 * sizeof(int)));
  multiply_nums<<<calculate_num_blocks(BLOCK_SIZE, n2), BLOCK_SIZE>>>(n2, total_p2_dev);

  int total_p1, total_p2;
  error_check(cudaMemcpy(&total_p1, total_p1_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaMemcpy(&total_p2, total_p2_dev, sizeof(int), cudaMemcpyDeviceToHost));
  printf("%d\n%d\n", total_p1, total_p2);

  error_check(cudaFree(total_p1_dev));
  error_check(cudaFree(total_p2_dev));

  return 0;
}
