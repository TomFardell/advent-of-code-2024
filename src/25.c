#include <stdio.h>

#define P

#ifdef P
#define FILE_NAME "data/25p.txt"
#define N 500
#endif
#ifdef S
#define FILE_NAME "data/25s.txt"
#define N 5
#endif

#define WIDTH 5
#define HEIGHT 7

int main(void) {
  int locks[WIDTH * N];
  int keys[WIDTH * N];
  int lock_p = 0;
  int key_p = 0;

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < N; i++) {
    char buffer[WIDTH * HEIGHT];
    for (int j = 0; j < HEIGHT; j++) {
      for (int k = 0; k < WIDTH; k++) {
        buffer[k + WIDTH * j] = fgetc(file);
      }
      fgetc(file);  // Read newline
      fgetc(file);
    }
    fgetc(file);
    fgetc(file);

    for (int j = 0; j < WIDTH; j++) {
      int count = 0;
      for (int k = 1; k < HEIGHT - 1; k++) {
        count += (buffer[j + WIDTH * k] == '#');
      }

      if (buffer[0] == '#') {
        locks[lock_p++] = count;
      } else {
        keys[key_p++] = count;
      }
    }
  }
  fclose(file);

  int n_locks = lock_p / WIDTH;
  int n_keys = key_p / WIDTH;

  int count = 0;

  for (int key = 0; key < n_keys; key++) {
    for (int lock = 0; lock < n_locks; lock++) {
      int match = 1;

      for (int i = 0; i < WIDTH; i++) {
        if (keys[i + WIDTH * key] + locks[i + WIDTH * lock] > HEIGHT - 2) {
          match = 0;
          break;
        }
      }

      count += match;
    }
  }

  printf("%d\n", count);

  return 0;
}
