#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.cuh"

void process_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "GPU error\n%d %s: %s\n", err, cudaGetErrorName(err), cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}