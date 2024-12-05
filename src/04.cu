#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "utils.cuh"

#define FILE_NAME "data/04p.txt"
#define N 140
#define BLOCK_SIZE 8

#define NUM_LINES (6 * N - 2)            // Number of distinct search lines
#define NUM_CROSSES ((N - 2) * (N - 2))  // Number of distinct crosses

// Don't need to pass line length since every line is length N
__device__ int count_words_on_line(const char *line, const char *str, int str_len) {
  int count = 0;
  int w = -1;

  for (int i = 0; i < N; i++) {
    if (line[i] == str[0]) {
      w = i;
    }
    if ((w != -1) && (line[i] != str[i - w])) {
      w = -1;
    }
    if ((w != -1) && i - w == str_len - 1) {
      count++;
      w = -1;
    }
  }

  return count;
}

__device__ int is_valid_cross(const char *cross) {
  if (cross[0] != 'A') return 0;
  int m_count = 0, s_count = 0;
  for (int i = 1; i < 5; i++) {
    switch (cross[i]) {
      case 'M': {
        m_count++;
      } break;
      case 'S': {
        s_count++;
      } break;
    }
  }

  return (m_count == 2 && s_count == 2 && cross[1] != cross[2] && cross[3] != cross[4]);
}

__global__ void count_words(char *lines, int *count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NUM_LINES) return;

  __shared__ int block_count;
  block_count = 0;
  __syncthreads();

  atomicAdd(&block_count, count_words_on_line(lines + i * N, "XMAS", 4));
  atomicAdd(&block_count, count_words_on_line(lines + i * N, "SAMX", 4));
  __syncthreads();

  if (threadIdx.x == 0) atomicAdd(count, block_count);
}

__global__ void count_crosses(char *crosses, int *count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (N - 2) * (N - 2)) return;

  __shared__ int block_count;
  block_count = 0;
  __syncthreads();

  atomicAdd(&block_count, is_valid_cross(crosses + i * 5));
  __syncthreads();

  if (threadIdx.x == 0) atomicAdd(count, block_count);
}

int main(void) {
  char line_buffer[N + 2];
  char grid[N][N];
  char lines[NUM_LINES][N] = {0};  // N rows/cols, 2N - 1 of each diagonal orientation
  char crosses[NUM_CROSSES][5];

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < N; i++) {
    fgets(line_buffer, N + 2, file);
    for (int j = 0; j < N; j++) {
      grid[i][j] = line_buffer[j];
    }
  }
  fclose(file);

  // Rows
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      lines[i][j] = grid[i][j];
    }
  }

  // Columns
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      lines[j + N][i] = grid[i][j];
    }
  }

  // Diagonals with constant sum
  for (int s = 0; s < 2 * N - 1; s++) {
    int l = s + 2 * N;  // 2N preceding lines
    if (s < N) {        // Sum to 0,1,...,N-1
      for (int i = 0; i <= s; i++) {
        lines[l][i] = grid[i][s - i];
      }
    } else {  // Sum to N,N+1,...,2N-2
      for (int i = 0; i < 2 * N - s - 1; i++) {
        lines[l][i] = grid[s - N + 1 + i][N - 1 - i];
      }
    }
  }

  // Diagonals with constant difference
  for (int d = 1 - N; d < N; d++) {
    int l = d + 5 * N - 2;  // 4N - 1 preceding lines, but d starts at 1 - N
    if (d > 0) {            // First element is strictly larger
      for (int i = 0; i < N - d; i++) {
        lines[l][i] = grid[d + i][i];
      }
    } else {  // Second element is larger or equal
      for (int i = 0; i < N + d; i++) {
        lines[l][i] = grid[i][i - d];
      }
    }
  }

  // Crosses
  for (int i = 0; i < N - 2; i++) {
    for (int j = 0; j < N - 2; j++) {
      crosses[(N - 2) * i + j][0] = grid[i + 1][j + 1];  // This must be an A
      // These two should not be equal
      crosses[(N - 2) * i + j][1] = grid[i][j];
      crosses[(N - 2) * i + j][2] = grid[i + 2][j + 2];
      // These two should not be equal
      crosses[(N - 2) * i + j][3] = grid[i][j + 2];
      crosses[(N - 2) * i + j][4] = grid[i + 2][j];
    }
  }

  int *word_count_dev, *crosses_count_dev;
  error_check(cudaMalloc(&word_count_dev, sizeof(int)));
  error_check(cudaMemset(word_count_dev, 0, sizeof(int)));
  error_check(cudaMalloc(&crosses_count_dev, sizeof(int)));
  error_check(cudaMemset(crosses_count_dev, 0, sizeof(int)));
  char *lines_dev;
  error_check(cudaMalloc(&lines_dev, NUM_LINES * N * sizeof(char)));
  error_check(cudaMemcpy(lines_dev, *lines, NUM_LINES * N * sizeof(char), cudaMemcpyHostToDevice));
  char *crosses_dev;
  error_check(cudaMalloc(&crosses_dev, NUM_CROSSES * 5 * sizeof(char)));
  error_check(
      cudaMemcpy(crosses_dev, *crosses, NUM_CROSSES * 5 * sizeof(char), cudaMemcpyHostToDevice));

  count_words<<<calculate_num_blocks(BLOCK_SIZE, NUM_LINES), BLOCK_SIZE>>>(lines_dev,
                                                                           word_count_dev);
  count_crosses<<<calculate_num_blocks(BLOCK_SIZE, NUM_CROSSES), BLOCK_SIZE>>>(crosses_dev,
                                                                               crosses_count_dev);

  error_check(cudaFree(lines_dev));
  error_check(cudaFree(crosses_dev));

  int word_count, crosses_count;
  error_check(cudaMemcpy(&word_count, word_count_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaFree(word_count_dev));
  error_check(cudaMemcpy(&crosses_count, crosses_count_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaFree(crosses_count_dev));

  printf("%d\n%d\n", word_count, crosses_count);

  return 0;
}