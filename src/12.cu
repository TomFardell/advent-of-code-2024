#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"

#define INPUT 'p'

#if INPUT == 'p'
#define FILE_NAME "data/12p.txt"
#define N 140
#elif INPUT == 's'
#define FILE_NAME "data/12s.txt"
#define N 10
#endif

#define BLOCK_SIZE 8

// Rectangle containing the region and indicator stating whether a given square is in the region
typedef struct {
  int i, j;          // Origin coordinates
  int i_len, j_len;  // Side lengths

  // Pointer to an indicator map where indicator[k + i_len * l] == 1 <=> (i + k, j + l) is a point
  // in the region (i.e. indexing is relative to the origin coordinates)
  char *indicator;
} region;

// Indicator with bounds check
__device__ char ind(const int i, const int j, const int i_len, const int j_len,
                    const char *indicator) {
  if (i < 0 || i_len <= i || j < 0 || j_len <= j) return 0;
  return indicator[i + i_len * j];
}

__global__ void sum_costs(const region *regions, const int n_regions, int *total) {
  int n = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (n >= n_regions) return;
  int area = 0;
  int sides = 0;
  int i_len = regions[n].i_len;
  int j_len = regions[n].j_len;
  char *indicator = regions[n].indicator;

  // First traverse in row-major order and check for sides above and below
  for (int i = 0; i < i_len; i++) {
    char u_side_on = 0;
    char d_side_on = 0;
    for (int j = 0; j < j_len; j++) {
      char u_ind = ind(i - 1, j, i_len, j_len, indicator);
      char d_ind = ind(i + 1, j, i_len, j_len, indicator);
      char c_ind = indicator[i + i_len * j];

      if (c_ind) area++;  // Also compute area on this traversal

      // If not currently making a side and empty space above, starting making a side
      if (!u_side_on && !u_ind && c_ind) {
        sides++;
        u_side_on = 1;
      }
      // If this side is broken, stop making it
      if (u_side_on && !(!u_ind && c_ind)) {
        u_side_on = 0;
      }

      if (!d_side_on && !d_ind && c_ind) {
        sides++;
        d_side_on = 1;
      }
      if (d_side_on && !(!d_ind && c_ind)) {
        d_side_on = 0;
      }
    }
  }

  // Repeat for sides left and right, traversing in column-major order
  for (int j = 0; j < j_len; j++) {
    char l_side_on = 0;
    char r_side_on = 0;
    for (int i = 0; i < i_len; i++) {
      char l_ind = ind(i, j - 1, i_len, j_len, indicator);
      char r_ind = ind(i, j + 1, i_len, j_len, indicator);
      char c_ind = indicator[i + i_len * j];

      if (!l_side_on && !l_ind && c_ind) {
        sides++;
        l_side_on = 1;
      }
      if (l_side_on && !(!l_ind && c_ind)) {
        l_side_on = 0;
      }

      if (!r_side_on && !r_ind && c_ind) {
        sides++;
        r_side_on = 1;
      }
      if (r_side_on && !(!r_ind && c_ind)) {
        r_side_on = 0;
      }
    }
  }

  atomicAdd(total, area * sides);
}

// Copy regions to device, freeing the host indicators while doing so
void copy_regions_to_device(region *regions, region *regions_dev, const int n_regions) {
  // For each region, allocate the indicator and copy the data across
  for (int n = 0; n < n_regions; n++) {
    char *pointer_on_host_to_indicator_on_device;
    error_check(cudaMalloc(&pointer_on_host_to_indicator_on_device,
                           regions[n].i_len * regions[n].j_len * sizeof(char)));
    error_check(cudaMemcpy(&(regions_dev[n].indicator), &pointer_on_host_to_indicator_on_device,
                           sizeof(char *), cudaMemcpyHostToDevice));
    error_check(
        cudaMemcpy(&(regions_dev[n].i), &(regions[n].i), sizeof(int), cudaMemcpyHostToDevice));
    error_check(
        cudaMemcpy(&(regions_dev[n].j), &(regions[n].j), sizeof(int), cudaMemcpyHostToDevice));
    error_check(cudaMemcpy(&(regions_dev[n].i_len), &(regions[n].i_len), sizeof(int),
                           cudaMemcpyHostToDevice));
    error_check(cudaMemcpy(&(regions_dev[n].j_len), &(regions[n].j_len), sizeof(int),
                           cudaMemcpyHostToDevice));
    error_check(cudaMemcpy(pointer_on_host_to_indicator_on_device, regions[n].indicator,
                           regions[n].i_len * regions[n].j_len * sizeof(char),
                           cudaMemcpyHostToDevice));
    free(regions[n].indicator);
  }
}

// Free indicators in regions_dev
void free_indicators_on_device(region *regions_dev, const int n_regions) {
  for (int n = 0; n < n_regions; n++) {
    char *pointer_on_host_to_indicator_on_device;
    error_check(cudaMemcpy(&pointer_on_host_to_indicator_on_device, &(regions_dev[n].indicator),
                           sizeof(char *), cudaMemcpyDeviceToHost));
    error_check(cudaFree(pointer_on_host_to_indicator_on_device));
  }
}

int in_grid(const int i, const int j) { return 0 <= i && i < N && 0 <= j && j < N; }

int main(void) {
  const int dirs[8] = {-1, 0, 1, 0, 0, 1, 0, -1};

  char map[N * N];
  char visited[N * N] = {0};
  char visited_prev[N * N] = {0};
  // This will be way too large for most inputs. Only used part of the array will be sent to device
  region regions[N * N];

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      fscanf(file, "%c", map + (i + N * j));
    }
    fscanf(file, "\n");  // Read newline
  }
  fclose(file);

  int cost = 0;
  int n_regions = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (visited[i + N * j]) continue;
      char plant = map[i + N * j];
      int area = 0, perim = 0;
      int i_min = N - 1, i_max = 0;
      int j_min = N - 1, j_max = 0;

      int stack[2 * N * N];
      int sp = 0;
      stack[0] = i;
      stack[1] = j;

      // For each adjacent sqaure...
      while (sp >= 0) {
        int si = stack[sp];
        int sj = stack[sp + 1];
        sp -= 2;

        if (visited[si + N * sj]) continue;

        // Update bounds for this region
        i_min = (si < i_min) ? si : i_min;
        i_max = (si > i_max) ? si : i_max;
        j_min = (sj < j_min) ? sj : j_min;
        j_max = (sj > j_max) ? sj : j_max;

        visited[si + N * sj] = 1;
        area++;
        perim += 4;

        // Check each adjacent sqaure of the same plant type. If already visited, decrement the
        // perimeter.  Otherwise, add to the stack
        for (int d = 0; d < 8; d += 2) {
          int ai = si + dirs[d];
          int aj = sj + dirs[d + 1];
          if (!in_grid(ai, aj) || map[ai + N * aj] != plant) continue;

          if (visited[ai + N * aj]) {
            perim -= 2;
          } else {
            sp += 2;
            stack[sp] = ai;
            stack[sp + 1] = aj;
          }
        }
      }

      cost += area * perim;

      // Add this region to the array
      regions[n_regions].i = i_min;
      regions[n_regions].j = j_min;
      regions[n_regions].i_len = i_max - i_min + 1;
      regions[n_regions].j_len = j_max - j_min + 1;
      regions[n_regions].indicator =
          (char *)malloc(regions[n_regions].j_len * regions[n_regions].i_len * sizeof(char));

      // Fill the indicator grid
      for (int i = i_min; i <= i_max; i++) {
        for (int j = j_min; j <= j_max; j++) {
          // 1 if this square was visited when traversing this region, 0 otherwise
          regions[n_regions].indicator[(i - i_min) + regions[n_regions].i_len * (j - j_min)] =
              (visited[i + N * j] > visited_prev[i + N * j]);
          // Update this square in visited_prev
          visited_prev[i + N * j] = visited[i + N * j];
        }
      }

      n_regions++;
    }
  }

  printf("%d\n", cost);

  region *regions_dev;
  error_check(cudaMalloc(&regions_dev, n_regions * sizeof(region)));
  copy_regions_to_device(regions, regions_dev, n_regions);

  int *total_dev;
  error_check(cudaMalloc(&total_dev, sizeof(int)));
  error_check(cudaMemset(total_dev, 0, sizeof(int)));

  sum_costs<<<calculate_num_blocks(BLOCK_SIZE, n_regions), BLOCK_SIZE>>>(regions_dev, n_regions,
                                                                         total_dev);
  free_indicators_on_device(regions_dev, n_regions);

  error_check(cudaMemcpy(&cost, total_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaFree(total_dev));

  printf("%d\n", cost);

  return 0;
}
