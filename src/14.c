//=====| NO PARALLELISM FOR THIS PUZZLE |===========================================================
// The first part of this puzzle was just performing a few caluclations for each robot, so there
// wouldn't be much point in writing a kernel, and I couldn't be bothered, so I didn't. Since the
// puzzle doesn't actually tell you what the Christmas tree looks like in part 2, I couldn't think
// of a better way than just printing out all the robot maps to the terminal until I caught a
// glimpse of something resembling a tree. This worked, but I didn't find the first tree. Now that I
// knew what the tree was supposed to look like, I sent the output into a text file and searched for
// XXXX..., until it found the upper border of the first Christmas tree image. I suppose I could
// have written a kernel to search for lines of consecutive robots in the maps, but - as mentioned
// before - I couldn't be bothered.
//==================================================================================================
#include <stdio.h>

#define INPUT 'p'

#if INPUT == 'p'
#define FILE_NAME "data/14p.txt"
#define N 500
#define X 101
#define Y 103
#elif INPUT == 's'
#define FILE_NAME "data/14s.txt"
#define N 12
#define X 11
#define Y 7
#endif

#define N_SECONDS 100
#define MAX_SEARCH 10000  // Initially I had this set to 100000
#define PRINT_MAPS 0      // Change to 1 to print out maps for finding the Christmas tree in part 2

typedef struct {
  int x, y;
} vec2;

typedef struct {
  vec2 p;
  vec2 v;
} robot;

int main() {
  robot robots[N];
  int score = 1;

  FILE *file = fopen(FILE_NAME, "r");
  for (int n = 0; n < N; n++) {
    fscanf(file, "p=%d,%d v=%d,%d\n", &robots[n].p.x, &robots[n].p.y, &robots[n].v.x,
           &robots[n].v.y);
  }
  fclose(file);

  for (int s = 0; s < MAX_SEARCH; s++) {
    int quad_counts[4] = {0};
    char map[X * Y] = {0};  // Indicator for robot positions after s seconds

    for (int n = 0; n < N; n++) {
      // Two modulos are needed to avoid negative x, y
      int x = ((robots[n].p.x + s * robots[n].v.x) % X + X) % X;
      int y = ((robots[n].p.y + s * robots[n].v.y) % Y + Y) % Y;

      if (x < X / 2 && y < Y / 2) quad_counts[0]++;
      if (x < X / 2 && y > Y / 2) quad_counts[1]++;
      if (x > X / 2 && y < Y / 2) quad_counts[2]++;
      if (x > X / 2 && y > Y / 2) quad_counts[3]++;

      map[x + X * y] = 1;
    }

    // Part 1
    if (s == 100) {
      for (int i = 0; i < 4; i++) {
        score *= quad_counts[i];
      }
    }

    // Print maps for finding the Christmas tree
    if (PRINT_MAPS) {
      for (int y = 0; y < Y; y++) {
        for (int x = 0; x < X; x++) {
          printf("%c", map[x + X * y] ? 'X' : ' ');
        }
        printf("\n");
      }
      printf("s = %d\n\n", s);
    }
  }

  printf("%d\n", score);

  return 0;
}
