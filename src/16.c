//=====| NO PARALLELISM FOR THIS PUZZLE |==========================================================
// Pathfinding is hard to parallelise, and I have been enjoying coding these in just C without
// having to deal with long CUDA calls everywhere.
//=================================================================================================
#include <stdio.h>
#include <limits.h>

#define P

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

typedef struct {
  int score, d, x, y;
} node;

v2 dirs[4] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};

void dijkstra(const char *map, const v2 start_pos, const int face_dir_idx, int *score_map) {
  char visited[4 * N * N] = {0};
  node stack[4 * N * N];
  int sp = 0;

  for (int i = 0; i < 4 * N * N; i++) {
    score_map[i] = INT_MAX;
  }
  stack[0] = (node){0, face_dir_idx, start_pos.x, start_pos.y};

  while (sp >= 0) {
    node this_node = stack[sp--];

    if (visited[this_node.d + 4 * (this_node.x + N * this_node.y)]) continue;

    int node_score = this_node.score;
    visited[this_node.d + 4 * (this_node.x + N * this_node.y)] = 1;  // Set this node as visited
    score_map[this_node.d + 4 * (this_node.x + N * this_node.y)] = node_score;  // Update the map

    // Rotation neighbours and then the move forwards neighbour
    node neighbours[3] = {{node_score + 1000, (this_node.d + 1) % 4, this_node.x, this_node.y},
                          {node_score + 1000, (this_node.d + 3) % 4, this_node.x, this_node.y},
                          {node_score + 1, this_node.d, this_node.x + dirs[this_node.d].x,
                           this_node.y + dirs[this_node.d].y}};

    for (int i = 0; i < 3; i++) {
      int n_score = neighbours[i].score;
      int n_d = neighbours[i].d;
      int n_x = neighbours[i].x;
      int n_y = neighbours[i].y;
      int c_score = score_map[n_d + 4 * (n_x + N * n_y)];

      if (map[n_x + N * n_y] == '#') continue;
      if (n_score < c_score) {
        // Extend the stack with a dummy node that will be overwritten later
        stack[++sp] = (node){-1, 0, 0, 0};

        // Insert this node in the stack, maintaining sorting (linear)
        for (int si = 0; si <= sp; si++) {
          if (n_score >= stack[si].score) {
            // Move nodes up
            for (int sj = sp; sj > si; sj--) {
              stack[sj] = stack[sj - 1];
            }
            // Insert this node
            stack[si] = neighbours[i];
            break;
          }
        }
      }
    }
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
      if (map[x + N * y] == 'S') start_pos = (v2){x, y};
      if (map[x + N * y] == 'E') end_pos = (v2){x, y};
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
