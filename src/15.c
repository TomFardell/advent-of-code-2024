//=====| NO PARALLELISM FOR THIS PUZZLE |===========================================================
// Since there is just one robot, there was no scope for paralleism with this puzzle (except for
// summing coordinates at the end). The implementation here is a bit messy with some repeated code,
// but it seems to work.
//==================================================================================================
#include <stdio.h>
#include <stdlib.h>

#define INPUT 'p'

#if INPUT == 'p'
#define FILE_NAME "data/15p.txt"
#define N 50
#define N_LINES 20
#define LINE_LENGTH 1000
#elif INPUT == 's'
#define FILE_NAME "data/15s.txt"
#define N 10
#define N_LINES 10
#define LINE_LENGTH 70
#elif INPUT == 'd'
#define FILE_NAME "data/15d.txt"
#define N 7
#define N_LINES 1
#define LINE_LENGTH 7
#endif

#define QUEUE_SIZE (N * N)

typedef struct {
  int x, y;
} v2;

v2 char_to_dir(const char dir_char) {
  switch (dir_char) {
    case '^':
      return (v2){0, -1};
    case '>':
      return (v2){1, 0};
    case 'v':
      return (v2){0, 1};
    case '<':
      return (v2){-1, 0};
    default:
      printf("Invalid character '%c'\n", dir_char);
      exit(EXIT_FAILURE);
  }
}

int main(void) {
  char warehouse[N * N];
  char wide_warehouse[2 * N * N];
  char instructions[N_LINES * LINE_LENGTH];
  v2 robot_pos;
  v2 robot_wide_pos;

  FILE *file = fopen(FILE_NAME, "r");
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x++) {
      fscanf(file, "%c", warehouse + x + N * y);

      if (warehouse[x + N * y] == '@') {
        robot_pos.x = x;
        robot_pos.y = y;
        robot_wide_pos.x = 2 * x;
        robot_wide_pos.y = y;
        wide_warehouse[(2 * x) + (2 * N) * y] = '@';
        wide_warehouse[(2 * x + 1) + (2 * N) * y] = '.';
      } else if (warehouse[x + N * y] == 'O') {
        wide_warehouse[(2 * x) + (2 * N) * y] = '[';
        wide_warehouse[(2 * x + 1) + (2 * N) * y] = ']';
      } else {
        wide_warehouse[(2 * x) + (2 * N) * y] = warehouse[x + N * y];
        wide_warehouse[(2 * x + 1) + (2 * N) * y] = warehouse[x + N * y];
      }
    }
    fscanf(file, "\n");
  }
  fscanf(file, "\n");
  for (int l = 0; l < N_LINES; l++) {
    for (int i = 0; i < LINE_LENGTH; i++) {
      fscanf(file, "%c", instructions + i + LINE_LENGTH * l);
    }
    fscanf(file, "\n");
  }
  fclose(file);

  for (int i = 0; i < N_LINES * LINE_LENGTH; i++) {
    v2 dir = char_to_dir(instructions[i]);
    v2 scan_pos = robot_pos;
    char scan_char;

    do {
      scan_pos.x += dir.x;
      scan_pos.y += dir.y;
      scan_char = warehouse[scan_pos.x + N * scan_pos.y];
    } while (scan_char != '#' && scan_char != '.');

    if (scan_char == '.') {
      for (int x = scan_pos.x, y = scan_pos.y; x != robot_pos.x || y != robot_pos.y;
           x -= dir.x, y -= dir.y) {
        warehouse[x + N * y] = warehouse[(x - dir.x + N * (y - dir.y))];
      }

      warehouse[robot_pos.x + N * robot_pos.y] = '.';
      robot_pos.x += dir.x;
      robot_pos.y += dir.y;
    }
  }

  for (int i = 0; i < N_LINES * LINE_LENGTH; i++) {
    v2 dir = char_to_dir(instructions[i]);
    v2 scan_pos = robot_wide_pos;
    char scan_char;

    if (dir.y == 0) {  // If moving horizontal proceed as before
      do {
        scan_pos.x += dir.x;
        scan_char = wide_warehouse[scan_pos.x + (2 * N) * scan_pos.y];
      } while (scan_char != '#' && scan_char != '.');

      if (scan_char == '.') {
        for (int x = scan_pos.x; x != robot_wide_pos.x; x -= dir.x) {
          wide_warehouse[x + 2 * N * robot_wide_pos.y] =
              wide_warehouse[x - dir.x + 2 * N * robot_wide_pos.y];
        }

        wide_warehouse[robot_wide_pos.x + 2 * N * robot_wide_pos.y] = '.';
        robot_wide_pos.x += dir.x;
      }
    } else {  // Otherwise use a queue to check layers of boxes
      int can_push = 1;
      v2 push_queue[QUEUE_SIZE];
      int hp = 0, tp = 1;
      push_queue[0] = robot_wide_pos;

      while (hp < tp) {
        v2 push_pos = push_queue[hp++];
        char push_char = wide_warehouse[push_pos.x + (2 * N) * (push_pos.y + dir.y)];

        if (push_char == '[') {
          v2 left_push_pos = {push_pos.x, push_pos.y + dir.y};
          push_queue[tp++] = left_push_pos;
          v2 right_push_pos = {push_pos.x + 1, push_pos.y + dir.y};
          push_queue[tp++] = right_push_pos;
        } else if (push_char == ']') {
          v2 left_push_pos = {push_pos.x - 1, push_pos.y + dir.y};
          push_queue[tp++] = left_push_pos;
          v2 right_push_pos = {push_pos.x, push_pos.y + dir.y};
          push_queue[tp++] = right_push_pos;
        } else if (push_char == '#') {
          can_push = 0;
          break;
        }
      }

      if (can_push) {
        // Go through the boxes and move them forwards
        for (int p = tp - 1; p > 0; p--) {
          v2 box_pos = push_queue[p];
          // Boxes could be in the queue twice, so need to check we are not moving them twice
          if (wide_warehouse[box_pos.x + (2 * N) * box_pos.y] != '.')
            wide_warehouse[box_pos.x + (2 * N) * (box_pos.y + dir.y)] =
                wide_warehouse[box_pos.x + (2 * N) * box_pos.y];
          wide_warehouse[box_pos.x + (2 * N) * box_pos.y] = '.';
        }

        wide_warehouse[robot_wide_pos.x + (2 * N) * robot_wide_pos.y] = '.';
        robot_wide_pos.y += dir.y;
        wide_warehouse[robot_wide_pos.x + (2 * N) * robot_wide_pos.y] = '@';
      }
    }
  }

  int total1 = 0;
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x++) {
      if (warehouse[x + N * y] == 'O') total1 += 100 * y + x;
    }
  }
  int total2 = 0;
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < 2 * N; x++) {
      if (wide_warehouse[x + (2 * N) * y] == '[') total2 += 100 * y + x;
    }
  }
  printf("%d\n%d\n", total1, total2);

  return 0;
}
