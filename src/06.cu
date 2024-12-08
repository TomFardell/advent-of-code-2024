#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "utils.cuh"

#define INPUT 'p'

#if INPUT == 'p'
#define FILE_NAME "data/06p.txt"
#define N 130
#elif INPUT == 's'
#define FILE_NAME "data/06s.txt"
#define N 10
#endif

#define BLOCK_SIZE 16

enum direction { UP, RIGHT, DOWN, LEFT };
enum grid_info { UNVISITED, VISITED, WALL };

typedef struct {
  int row, col;
} point;

typedef struct {
  point p;
  direction d;
} grid_square;

__constant__ int N_obs_dev;
__constant__ grid_square start_square_dev;

// Get change in coordinates associated with a given direction
__host__ __device__ grid_square convert_direction(direction d) {
  grid_square change = {0, 0, d};

  switch (d) {
    case UP:
      change.p.row = -1;
      break;
    case RIGHT:
      change.p.col = 1;
      break;
    case DOWN:
      change.p.row = 1;
      break;
    case LEFT:
      change.p.col = -1;
      break;
  }

  return change;
}

__host__ __device__ int square_in_bounds(const point square) {
  return 0 <= square.row && square.row < N && 0 <= square.col && square.col < N;
}

__host__ __device__ grid_square get_next_square(const grid_square square, const char *grid) {
  grid_square move = convert_direction(square.d);
  grid_square next_square = {.p = {square.p.row + move.p.row, square.p.col + move.p.col},
                             .d = square.d};

  if (square_in_bounds(next_square.p) &&
      grid[(next_square.p.row * N) + next_square.p.col] == WALL) {
    next_square.p.row = square.p.row;
    next_square.p.col = square.p.col;
    next_square.d = (direction)((square.d + 1) % 4);
  }

  return next_square;
}

// Modifies grid to contain visited squares for use in second part
int count_unique_squares_visited(const grid_square start_square, char *grid) {
  int count = 0;

  // Iterate through squares until square is out of bounds
  for (grid_square sq = start_square; square_in_bounds(sq.p); sq = get_next_square(sq, grid)) {
    if (grid[(sq.p.row * N) + sq.p.col] == UNVISITED) {
      grid[(sq.p.row * N) + sq.p.col] = VISITED;
      count++;
    }
  }

  return count;
}

__device__ int has_infinite_loop(const char *grid, const point obs) {
  char dir_grid[N * N * 4] = {UNVISITED};
  char new_grid[N * N] = {UNVISITED};

  // Construct the grid with new obstacle added and all squares unvisited
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      // If this square has a wall in the original grid or is the obstacle square, add a wall in
      // dir_grid
      if (grid[(row * N) + col] == WALL || (obs.row == row && obs.col == col)) {
        dir_grid[(((row * N) + col) * 4) + UP] = WALL;
        dir_grid[(((row * N) + col) * 4) + RIGHT] = WALL;
        dir_grid[(((row * N) + col) * 4) + DOWN] = WALL;
        dir_grid[(((row * N) + col) * 4) + LEFT] = WALL;
        new_grid[(row * N) + col] = WALL;
      }
    }
  }

  grid_square sq;
  for (sq = start_square_dev; square_in_bounds(sq.p); sq = get_next_square(sq, new_grid)) {
    // If this square has already been visited facing this way, return 1
    int index = (((sq.p.row * N) + sq.p.col) * 4) + sq.d;
    if (dir_grid[index] == VISITED) {
      return 1;
    }

    dir_grid[index] = VISITED;
  }

  return 0;
}

__global__ void count_loop_obstacles(const char *grid, const point *obs_pos, int *total) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= N_obs_dev) return;

  atomicAdd(total, has_infinite_loop(grid, obs_pos[i]));
}

int main(void) {
  char grid[N * N] = {UNVISITED};  // Storing as char to ensure 1 byte (so fits in constant memory)
  grid_square start_square;

  FILE *file = fopen(FILE_NAME, "r");
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      char c = fgetc(file);
      switch (c) {
        case '#':
          grid[(row * N) + col] = WALL;
          break;
        case '^':
          start_square.p.row = row;
          start_square.p.col = col;
          start_square.d = UP;
          break;
      }
    }
    fgetc(file);  // Read newline
  }
  fclose(file);

  int squares_visited = count_unique_squares_visited(start_square, grid);
  // -1 since we don't consider the guard's starting position
  int N_obs = squares_visited - 1;

  point *obs_pos = (point *)malloc(N_obs * sizeof(grid_square));  // Cast needed in C++

  int c = 0;
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      if (grid[(row * N) + col] == VISITED &&
          (row != start_square.p.row || col != start_square.p.col)) {
        obs_pos[c].row = row;
        obs_pos[c].col = col;
        c++;  // !!!
      }
    }
  }

  error_check(cudaMemcpyToSymbol(N_obs_dev, &N_obs, sizeof(int)));
  error_check(cudaMemcpyToSymbol(start_square_dev, &start_square, sizeof(grid_square)));

  char *grid_dev;
  error_check(cudaMalloc(&grid_dev, N * N * sizeof(char)));
  error_check(cudaMemcpy(grid_dev, grid, N * N * sizeof(char), cudaMemcpyHostToDevice));
  point *obs_pos_dev;
  error_check(cudaMalloc(&obs_pos_dev, N_obs * sizeof(point)));
  error_check(cudaMemcpy(obs_pos_dev, obs_pos, N_obs * sizeof(point), cudaMemcpyHostToDevice));
  free(obs_pos);
  int *total_dev;
  error_check(cudaMalloc(&total_dev, sizeof(int)));
  error_check(cudaMemset(total_dev, 0, sizeof(int)));

  count_loop_obstacles<<<calculate_num_blocks(BLOCK_SIZE, N_obs), BLOCK_SIZE>>>(
      grid_dev, obs_pos_dev, total_dev);

  error_check(cudaFree(grid_dev));
  error_check(cudaFree(obs_pos_dev));

  int total;
  error_check(cudaMemcpy(&total, total_dev, sizeof(int), cudaMemcpyDeviceToHost));
  error_check(cudaFree(total_dev));

  printf("%d\n%d\n", squares_visited, total);

  return 0;
}
