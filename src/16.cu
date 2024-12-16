//=====| NO PARALLELISM FOR THIS PUZZLE |==========================================================
// Pathfinding is hard to parallelise, and I have been enjoying coding these in just C without
// having to deal with long CUDA calls everywhere.
//=================================================================================================
#include <stdio.h>
#include <stdlib.h>

#define S1

#ifdef P
#define FILE_NAME "data/16p.txt"
#define N 141
#endif
#ifdef S1
#define FILE_NAME "data/16s1.txt"
#define N 15
#endif
#ifdef S2
#define FILE_NAME "data/16s2.txt"
#define N 17
#endif

typedef struct {
  int x, y;
} v2;

v2 dirs[4] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};

void dijkstra(const char *map, const v2 start_pos, const int face_dir_idx, int *score_map) {
  char visited[4 * N * N] = {0};

  for (int i = 0; i < 4 * N * N; i++) {
    score_map[i] = INT_MAX;
  }
  score_map[face_dir_idx + 4 * (start_pos.x + N * start_pos.y)] = 0;

  for (int node_count = 0; node_count < 4 * N * N; node_count++) {
    // First find the unvisited node with lowest score (naively since the grid isn't too large)
    int node_d, node_x, node_y;
    int min_score = INT_MAX;
    for (int d = 0; d < 4; d++) {
      for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
          // If not visited and has a lower score than the current minimum, set as new minimum
          if (!visited[d + 4 * (x + N * y)] && score_map[d + 4 * (x + N * y)] < min_score) {
            min_score = score_map[d + 4 * (x + N * y)];
            node_d = d;
            node_x = x;
            node_y = y;
          }
        }
      }
    }

    int node_score = score_map[node_d + 4 * (node_x + N * node_y)];
    visited[node_d + 4 * (node_x + N * node_y)] = 1;  // Set this node as visited

    // Go through neighbours and update their scores
    int rotation_neighbours[2] = {
        ((node_d + 1) % 4) + 4 * (node_x + N * node_y),  // Clockwise rotation
        ((node_d + 3) % 4) + 4 * (node_x + N * node_y),  // Anticlockwise rotation
    };
    for (int i = 0; i < 2; i++) {
      int current_score = score_map[rotation_neighbours[i]];
      int potential_score = node_score + 1000;
      if (potential_score < current_score) score_map[rotation_neighbours[i]] = potential_score;
    }
    int move_neighbour = node_d + 4 * ((node_x + dirs[node_d].x) + N * (node_y + dirs[node_d].y));
    // If the node in front is a wall, set it as visited and move on
    if (map[move_neighbour / 4] == '#') {
      visited[move_neighbour] = 1;
      node_count++;
      continue;
    }
    int current_score = score_map[move_neighbour];
    int potential_score = node_score + 1;
    if (potential_score < current_score) score_map[move_neighbour] = potential_score;
  }
}

int main(void) {
  char map[N * N];
  int score_map[4 * N * N];
  int reverse_score_map[4 * N * N];
  v2 start_pos, end_pos;

  FILE *file = fopen(FILE_NAME, "r");
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x++) {
      fscanf(file, "%c", map + x + N * y);
      if (map[x + N * y] == 'S') start_pos = {x, y};
      if (map[x + N * y] == 'E') end_pos = {x, y};
    }
    fscanf(file, "\n");
  }
  fclose(file);

  dijkstra(map, start_pos, 0, score_map);
  dijkstra(map, end_pos, 0, reverse_score_map);

  // Fill reverse_score_map with minimum score for all 4 facing directions of the endpoint
  int reverse_score_map_dir[4 * N * N];
  for (int d = 1; d < 4; d++) {
    dijkstra(map, end_pos, d, reverse_score_map_dir);
    for (int i = 0; i < 4 * N * N; i++) {
      if (reverse_score_map_dir[i] < reverse_score_map[i])
        reverse_score_map[i] = reverse_score_map_dir[i];
    }
  }

  int min_score = INT_MAX;
  for (int d = 0; d < 4; d++) {
    int this_score = score_map[d + 4 * (end_pos.x + N * end_pos.y)];
    if (this_score < min_score) min_score = this_score;
  }
  printf("%d\n", min_score);

  int count = 0;
  for (int i = 0; i < N * N; i++) {
    for (int d1 = 0; d1 < 4; d1++) {
      int d2 = (d1 + 2) % 4;
      if (score_map[d1 + 4 * i] + reverse_score_map[d2 + 4 * i] == min_score) {
        count++;
        map[i] = 'O';
        break;
      }
    }
  }
  printf("%d\n", count);

  return 0;
}
