#include <stdio.h>
#include <stdlib.h>

#define P

#ifdef P
#define FILE_NAME "data/18p.txt"
#define N 3450
#define N1 1024
#define G 71
#endif
#ifdef S
#define FILE_NAME "data/18s.txt"
#define N 25
#define N1 12
#define G 7
#endif

const int dirs[8] = {0, 1, 0, -1, 1, 0, -1, 0};

typedef struct {
  int x, y;
} v2;

int in_grid(const int x, const int y) { return (0 <= x && x < G && 0 <= y && y < G); }

// Returns steps to the end or -1 if the end is unreachable
int steps_to_end(const char *grid) {
  int distances[G * G] = {0};
  v2 queue[G * G];
  int hp = 0;
  int tp = 0;

  queue[hp++] = {0, 0};

  while (hp > tp) {
    v2 point = queue[tp++];
    for (int di = 0; di < 4; di++) {
      int nx = point.x + dirs[2 * di];
      int ny = point.y + dirs[2 * di + 1];

      if (in_grid(nx, ny) && grid[nx + G * ny] != 1) {
        // Don't add a node to the stack a second time
        if (distances[nx + G * ny] > 0) continue;

        // Otherwise, update the distance of this node
        int dist = distances[point.x + G * point.y] + 1;
        if (nx == G - 1 && ny == G - 1) return dist;
        distances[nx + G * ny] = dist;
        queue[hp++] = {nx, ny};
      }
    }
  }

  return -1;
}

int main(void) {
  FILE *file = fopen(FILE_NAME, "r");
  int points[2 * N];
  char grid[G * G] = {0};

  for (int i = 0; i < N; i++) {
    fscanf(file, "%d,%d", points + 2 * i, points + 2 * i + 1);
    fscanf(file, "\n");
  }
  fclose(file);

  for (int i = 0; i < N1; i++) {
    int x = points[2 * i];
    int y = points[2 * i + 1];
    grid[x + G * y] = 1;
  }

  printf("%d\n", steps_to_end(grid));

  // This is just bruteforce, but still runs very quickly
  for (int i = N1 + 1; i < N; i++) {
    int x = points[2 * i];
    int y = points[2 * i + 1];
    grid[x + G * y] = 1;
    int steps = steps_to_end(grid);
    if (steps == -1) {
      printf("%d,%d\n", x, y);
      break;
    }
  }

  return 0;
}
