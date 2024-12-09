#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "utils.cuh"

#define INPUT 'p'

#if INPUT == 'p'
#define FILE_NAME "data/09p.txt"
#define N 19999
#elif INPUT == 's'
#define FILE_NAME "data/09s.txt"
#define N 19
#elif INPUT == 'd'
#define FILE_NAME "data/09d.txt"
#define N 7
#endif

#define BLOCK_SIZE 64

typedef long long unsigned u64;

__global__ void calculate_checksum(const int *disk, const int disk_size, u64 *checksum) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= disk_size) return;

  __shared__ u64 block_sum;
  block_sum = 0;
  __syncthreads();

  atomicAdd(&block_sum, i * disk[i]);
  __syncthreads();

  if (threadIdx.x == 0) atomicAdd(checksum, block_sum);
}

int main(void) {
  char nums[N];

  FILE *file = fopen(FILE_NAME, "r");
  int disk_size = 0;
  int disk_defrag_size = 0;
  for (int l = 0; l < N; l++) {
    nums[l] = (int)(fgetc(file) - '0');
    if (l % 2 == 0) disk_size += nums[l];
    disk_defrag_size += nums[l];
  }
  fclose(file);

  int *disk = (int *)malloc(disk_size * sizeof(int));

  // Part 1 algorithm
  {
    int p1 = 0, p2 = N - 1;
    int id1 = 0, id2 = N / 2;
    int bc1 = 0, bc2 = 0;
    for (int dp = 0; dp < disk_size; dp++) {
      if (p1 % 2 == 0) {  // p1 points to data
        disk[dp] = id1;
        bc1++;
        // Once we have added nums[p1] id1s to disk, increment p1
        if (bc1 == nums[p1]) {
          bc1 = 0;
          id1++;
          p1++;
        }
      } else {  // p1 points to empty space
        // If we run out of empty space, increment p1
        if (bc1 == nums[p1]) {
          bc1 = 0;
          p1++;
          dp--;  // Since we are skipping the next part
          continue;
        }
        disk[dp] = id2;
        bc1++, bc2++;
        // If the block at the end runs out, decrement p2 (by 2)
        if (bc2 == nums[p2]) {
          bc2 = 0;
          id2--;
          p2 -= 2;
        }
      }
    }
  }

  int *disk_defrag = (int *)malloc(disk_defrag_size * sizeof(int));

  // Initialise disk in expanded form
  for (int i = 0, p = 0, id = 0; p < N; p++, id++) {
    for (int dp = 0; dp < nums[p]; dp++) {
      disk_defrag[i] = (p % 2 == 0) ? id / 2 : -1;
      i++;
    }
  }

  // Part 2 algorithm
  {
    for (int id = N / 2; id > 0; id--) {
      int p = 2 * id;
      int size = nums[p];
      int moved = 0;

      for (int bc = 0, dp = 0; dp < disk_defrag_size; dp++, bc++) {
        if (disk_defrag[dp] != -1) bc = 0;

        // When we find the original data...
        if (disk_defrag[dp] == id) {
          // If we moved the data, clear it here
          if (moved)
            for (int b = 0; b < size; b++) disk_defrag[dp + b] = -1;

          // In either case, stop traversing the disk
          break;
        }

        // If we are yet to move the data and a large enough block is found, move data here
        if (!moved && (bc == size)) {
          moved = 1;
          for (int b = 0; b < size; b++) disk_defrag[dp - b] = id;
        }
      }
    }
  }

  // Set -1 squares to 0 so checksum is calculated correctly
  for (int i = 0; i < disk_defrag_size; i++) {
    if (disk_defrag[i] == -1) disk_defrag[i] = 0;
  }

  int *disk_dev;
  error_check(cudaMalloc(&disk_dev, disk_size * sizeof(int)));
  error_check(cudaMemcpy(disk_dev, disk, disk_size * sizeof(int), cudaMemcpyHostToDevice));
  free(disk);
  int *disk_defrag_dev;
  error_check(cudaMalloc(&disk_defrag_dev, disk_defrag_size * sizeof(int)));
  error_check(cudaMemcpy(disk_defrag_dev, disk_defrag, disk_defrag_size * sizeof(int),
                         cudaMemcpyHostToDevice));
  free(disk_defrag);

  u64 *checksum_dev;
  error_check(cudaMalloc(&checksum_dev, sizeof(u64)));
  error_check(cudaMemset(checksum_dev, 0, sizeof(u64)));
  u64 *checksum_defrag_dev;
  error_check(cudaMalloc(&checksum_defrag_dev, sizeof(u64)));
  error_check(cudaMemset(checksum_defrag_dev, 0, sizeof(u64)));

  calculate_checksum<<<calculate_num_blocks(BLOCK_SIZE, disk_size), BLOCK_SIZE>>>(
      disk_dev, disk_size, checksum_dev);
  error_check(cudaPeekAtLastError());
  error_check(cudaFree(disk_dev));

  calculate_checksum<<<calculate_num_blocks(BLOCK_SIZE, disk_defrag_size), BLOCK_SIZE>>>(
      disk_defrag_dev, disk_defrag_size, checksum_defrag_dev);
  error_check(cudaPeekAtLastError());
  error_check(cudaFree(disk_defrag_dev));

  u64 checksum;
  error_check(cudaMemcpy(&checksum, checksum_dev, sizeof(u64), cudaMemcpyDeviceToHost));
  error_check(cudaFree(checksum_dev));

  u64 checksum_defrag;
  error_check(
      cudaMemcpy(&checksum_defrag, checksum_defrag_dev, sizeof(u64), cudaMemcpyDeviceToHost));
  error_check(cudaFree(checksum_defrag_dev));

  printf("%llu\n%llu\n", checksum, checksum_defrag);

  return 0;
}