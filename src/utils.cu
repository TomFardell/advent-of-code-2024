#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.cuh"

static void merge(int *arr, int m, int N) {
  int n1 = m;
  int n2 = N - m;

  int *L = (int *)malloc(n1 * sizeof(int));
  int *R = (int *)malloc(n2 * sizeof(int));
  if (!(L && R)) {
    fprintf(stderr, "Error allocating memory during merge sort\n");
    exit(EXIT_FAILURE);
  }

  /* Copy arr into temporary memory */
  for (int i1 = 0; i1 < n1; i1++) {
    L[i1] = arr[i1];
  }
  for (int i2 = 0; i2 < n2; i2++) {
    R[i2] = arr[m + i2];
  }

  int i1 = 0, i2 = 0;
  while (i1 < n1 && i2 < n2) {
    if (L[i1] < R[i2]) {
      arr[i1 + i2] = L[i1];
      i1++;
    } else {
      arr[i1 + i2] = R[i2];
      i2++;
    }
  }

  /* Append remaining elements to arr */
  while (i1 < n1) {
    arr[i1 + i2] = L[i1];
    i1++;
  }
  while (i2 < n2) {
    arr[i1 + i2] = R[i2];
    i2++;
  }

  free(L);
  free(R);
}

void merge_sort(int *arr, int N, int *res) {
  if (N == 0) return;
  if (N == 1) {
    res[0] = arr[0];
    return;
  }

  int m = N / 2;
  merge_sort(arr, m, res);
  merge_sort(arr + m, N - m, res + m);
  merge(res, m, N);
}

void print_int_array(int *arr, int N) {
  for (int i = 0; i < N; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

void error_check(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "GPU error\n%d %s: %s\n", err, cudaGetErrorName(err), cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}