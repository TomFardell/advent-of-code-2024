#include <stdio.h>
#include <stdlib.h>

#define P

#ifdef P
#define FILE_NAME "data/20p.txt"
#define N 141
#define M 100
#endif
#ifdef S
#define FILE_NAME "data/20s.txt"
#define N 15
#define M 50
#endif

typedef struct {
  int x, y;
} v2;

const v2 dirs[4] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

int in_grid(int x, int y) { return (0 <= x && x < N && 0 <= y && y < N); }

int main(void) {
  char map[N * N];
  FILE *file = fopen(FILE_NAME, "r");
  v2 end;

  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x++) {
      fscanf(file, "%c ", map + x + N * y);
      if (map[x + N * y] == 'E') end.x = x, end.y = y;
    }
  }

  // Initialise distances from end to -1
  int d_from_end[N * N];
  for (int i = 0; i < N * N; i++) d_from_end[i] = -1;

  v2 path[N * N] = {0};

  int dist = 0;
  v2 p = end;
  path[dist] = p;
  d_from_end[p.x + N * p.y] = dist++;  // Set end distance from end to 0

  // Compute path and distance lookup table
  while (map[p.x + N * p.y] != 'S') {
    for (int d = 0; d < 4; d++) {
      // Shouldn't need to bounds check as we will never be in any of the outer squares
      int nx = p.x + dirs[d].x;
      int ny = p.y + dirs[d].y;
      // If we are looking at an univisted empty space, this is the next square in the path
      if (map[nx + N * ny] != '#' && d_from_end[nx + N * ny] == -1) {
        path[dist].x = nx, path[dist].y = ny;
        d_from_end[nx + N * ny] = dist++;
        p.x = nx, p.y = ny;
        break;  // Since there is only one path, don't need to check any remaining neighbours
      }
    }
  }

  int path_size = dist;

  int count1 = 0;
  int count2 = 0;

  for (int i = 0; i < path_size; i++) {
    // Check shortcuts in each of the 4 directions for part 1
    for (int d = 0; d < 4; d++) {
      int sx = path[i].x + 2 * dirs[d].x;
      int sy = path[i].y + 2 * dirs[d].y;
      // If the shortcut destination is in the path
      if (in_grid(sx, sy) && d_from_end[sx + N * sy] != -1) {
        int time_saved = d_from_end[path[i].x + N * path[i].y] - d_from_end[sx + N * sy] - 2;
        if (time_saved >= M) count1++;
      }
    }

    // Check squares within 20 tiles for part 2
    for (int dx = -20; dx <= 20; dx++) {
      for (int dy = -20 + abs(dx); dy <= 20 - abs(dx); dy++) {
        int sx = path[i].x + dx;
        int sy = path[i].y + dy;
        // If the shortcut destination is in the path
        if (in_grid(sx, sy) && d_from_end[sx + N * sy] != -1) {
          int time_saved =
              d_from_end[path[i].x + N * path[i].y] - d_from_end[sx + N * sy] - abs(dx) - abs(dy);
          if (time_saved >= M) count2++;
        }
      }
    }
  }

  printf("%d\n%d\n", count1, count2);

  return 0;
}
