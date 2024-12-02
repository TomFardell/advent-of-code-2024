#ifndef __UTILS_H_
#define __UTILS_H_

#include <cuda.h>

void merge_sort(int *arr, int N, int *res);
void print_int_array(int *arr, int N);
void error_check(cudaError_t err);

#endif