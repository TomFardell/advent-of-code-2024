#ifndef __UTILS_H_
#define __UTILS_H_

#include <cuda.h>

void merge_sort(int *arr, int N, int *res);
void print_int_array(int *arr, int N);
int calculate_num_blocks(int block_size, int desired_threads);
void _error_check(const cudaError_t err, const int line, const char *file, const char *func);

#define error_check(err)                               \
  do {                                                 \
    _error_check((err), __LINE__, __FILE__, __func__); \
  } while (0);

#endif
