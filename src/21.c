#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define P

#ifdef P
#define FILE_NAME "data/21p.txt"
#define N 5
#endif
#ifdef S
#define FILE_NAME "data/21s.txt"
#define N 5
#endif

typedef long long ll;

typedef struct {
  ll costs[5 * 5];  // costs[h(a) + 5 * h(b)] holds the cost of pressing key a, starting at key b
} DirKeypad;

typedef struct {
  int x, y;
} v2;

int h(const char dir) {
  switch (dir) {
    case 'A':
      return 0;
    case '^':
      return 1;
    case '>':
      return 2;
    case 'v':
      return 3;
    case '<':
      return 4;
  }

  fprintf(stderr, "Character is not a direction.\n");
  exit(EXIT_FAILURE);
}

char dir_to_char(const v2 dir) {
  if (dir.x == -1 && dir.y == 0) return '<';
  if (dir.x == 1 && dir.y == 0) return '>';
  if (dir.x == 0 && dir.y == -1) return '^';
  if (dir.x == 0 && dir.y == 1) return 'v';

  return 'X';
}

v2 num_key_coords(const char key) {
  switch (key) {
    case 'A':
      return (v2){0, 0};
    case '0':
      return (v2){-1, 0};
    case '1':
      return (v2){-2, -1};
    case '2':
      return (v2){-1, -1};
    case '3':
      return (v2){0, -1};
    case '4':
      return (v2){-2, -2};
    case '5':
      return (v2){-1, -2};
    case '6':
      return (v2){0, -2};
    case '7':
      return (v2){-2, -3};
    case '8':
      return (v2){-1, -3};
    case '9':
      return (v2){0, -3};
  }

  fprintf(stderr, "Character is not a number.\n");
  exit(EXIT_FAILURE);
}

ll min(const ll a, const ll b) { return (a < b) ? a : b; }

int sign(const int x) { return (x > 0) - (x < 0); }

ll cost(const char *seq, const int n, const DirKeypad input) {
  ll total = 0;

  for (int i = 0; i < n; i++) {
    total += input.costs[h(seq[i]) + 5 * h(seq[(i + n - 1) % n])];
  }

  return total;
}

int in_numpad(v2 p) {
  return ((-2 <= p.x && p.x <= 0 && -3 <= p.y && p.y <= 0) && !(p.x == -2 && p.y == 0));
}

DirKeypad get_next_keypad(const DirKeypad input) {
  DirKeypad next_keypad;
  ll *c_n;

  c_n = next_keypad.costs + 5 * h('A');
  c_n[h('A')] = cost("A", 1, input);
  c_n[h('^')] = cost("<A", 2, input);
  c_n[h('>')] = cost("vA", 2, input);
  c_n[h('v')] = min(cost("<vA", 3, input), cost("v<A", 3, input));
  c_n[h('<')] = min(cost("v<<A", 4, input), cost("<v<A", 4, input));

  c_n = next_keypad.costs + 5 * h('^');
  c_n[h('A')] = cost(">A", 2, input);
  c_n[h('^')] = cost("A", 1, input);
  c_n[h('>')] = min(cost(">vA", 3, input), cost("v>A", 3, input));
  c_n[h('v')] = cost("vA", 2, input);
  c_n[h('<')] = cost("v<A", 3, input);

  c_n = next_keypad.costs + 5 * h('>');
  c_n[h('A')] = cost("^A", 2, input);
  c_n[h('^')] = min(cost("^<A", 3, input), cost("<^A", 3, input));
  c_n[h('>')] = cost("A", 1, input);
  c_n[h('v')] = cost("<A", 2, input);
  c_n[h('<')] = cost("<<A", 3, input);

  c_n = next_keypad.costs + 5 * h('v');
  c_n[h('A')] = min(cost(">^A", 3, input), cost("^>A", 3, input));
  c_n[h('^')] = cost("^A", 2, input);
  c_n[h('>')] = cost(">A", 2, input);
  c_n[h('v')] = cost("A", 1, input);
  c_n[h('<')] = cost("<A", 2, input);

  c_n = next_keypad.costs + 5 * h('<');
  c_n[h('A')] = min(cost(">>^A", 4, input), cost(">^>A", 4, input));
  c_n[h('^')] = cost(">^A", 3, input);
  c_n[h('>')] = cost(">>A", 3, input);
  c_n[h('v')] = cost(">A", 2, input);
  c_n[h('<')] = cost("A", 1, input);

  return next_keypad;
}

ll shortest_path(const char start, const char end, const DirKeypad input) {
  v2 s = num_key_coords(start);
  v2 e = num_key_coords(end);
  v2 diff = {e.x - s.x, e.y - s.y};

  v2 dx = {sign(diff.x), 0};
  v2 dy = {0, sign(diff.y)};
  char dx_c = dir_to_char(dx);
  char dy_c = dir_to_char(dy);
  int n = abs(diff.x) + abs(diff.y);

  ll shortest = LLONG_MAX;

  // p = 0, ..., 2^n - 1. Since there are only two directions can view them as binary strings
  for (int p = 0; p < (1 << n); p++) {
    int xc = 0, yc = 0;
    v2 pos = s;
    char path[6];  // Largest possible path is length 6
    int valid = 1;
    for (int i = 0; i < n; i++) {
      if ((p >> i) % 2 == 0) {
        path[i] = dx_c;
        xc++;
        pos.x += dx.x;
      } else {
        path[i] = dy_c;
        yc++;
        pos.y += dy.y;
      }

      // Give up on this path if it goes out the numpad or goes too far in a direction
      if (!in_numpad(pos) || xc > abs(diff.x) || yc > abs(diff.y)) {
        valid = 0;
        break;
      }
    }

    if (valid) {
      path[n] = 'A';
      shortest = min(shortest, cost(path, n + 1, input));
    }
  }

  return shortest;
}

ll code_cost(const char *seq, const int n, const DirKeypad input) {
  ll total = 0;

  for (int i = 0; i < n; i++) {
    total += shortest_path(seq[(i + n - 1) % n], seq[i], input);
  }

  return total;
}

int main(void) {
  char codes[4 * N];

  FILE *file = fopen(FILE_NAME, "r");
  for (int l = 0; l < N; l++) {
    for (int c = 0; c < 4; c++) {
      codes[c + 4 * l] = fgetc(file);
    }
    fgetc(file);  // Read empty space
    fgetc(file);  // Read newline
  }
  fclose(file);

  DirKeypad human_keypad;
  for (int i = 0; i < 5 * 5; i++) human_keypad.costs[i] = 1;

  DirKeypad num_input_keypad = get_next_keypad(get_next_keypad(human_keypad));
  DirKeypad far_keypad = human_keypad;
  for (int i = 0; i < 25; i++) {
    far_keypad = get_next_keypad(far_keypad);
  }

  ll complexity = 0;
  ll far_complexity = 0;
  for (int i = 0; i < N; i++) {
    char this_code[5] = {0};
    for (int c = 0; c < 3; c++) {
      this_code[c] = codes[c + 4 * i];
    }
    int code_val;
    sscanf(this_code, "%d", &code_val);
    this_code[3] = 'A';

    complexity += code_cost(this_code, 4, num_input_keypad) * code_val;
    far_complexity += code_cost(this_code, 4, far_keypad) * code_val;
  }

  printf("%lld\n%lld\n", complexity, far_complexity);

  return 0;
}
