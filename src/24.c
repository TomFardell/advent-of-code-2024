#include <stdio.h>
#include <stdlib.h>

#define P

#ifdef P
#define FILE_NAME "data/24p.txt"
#define N 90
#define M 222
#endif
#ifdef S
#define FILE_NAME "data/24s.txt"
#define N 10
#define M 36
#endif

#define A 36

typedef long long unsigned llu;
typedef enum { OR, AND, XOR } Operation;

typedef struct {
  Operation op;
  int in1, in2, out;
} Instruction;

int char_index(const char c) {
  if (c <= '9')
    return c - '0';
  else
    return 10 + c - 'a';
}

char char_index_inv(const int n) {
  if (n <= 9)
    return n + '0';
  else
    return n - 10 + 'a';
}

int h(const char *seq) {
  return char_index(seq[2]) + A * (char_index(seq[1]) + A * char_index(seq[0]));
}

void h_inv(const int n, char *s) {
  s[0] = char_index_inv(n / (A * A));
  s[1] = char_index_inv((n / A) % A);
  s[2] = char_index_inv(n % A);
  s[3] = '\0';
}

int main(void) {
  Instruction instructions[M];

  int *vals = malloc((sizeof *vals) * A * A * A);
  for (int i = 0; i < A * A * A; i++) {
    vals[i] = -1;
  }

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < N; i++) {
    char wire[4];
    fscanf(file, "%3s", wire);
    fscanf(file, ": %d ", vals + h(wire));
  }
  for (int i = 0; i < M; i++) {
    char in1[4], op[4], in2[4], out[4];
    fscanf(file, "%3s %3[^ ] %3s -> %3s ", in1, op, in2, out);
    instructions[i].in1 = h(in1);
    instructions[i].in2 = h(in2);
    instructions[i].out = h(out);
    if (op[0] == 'O' && op[1] == 'R')
      instructions[i].op = OR;
    else if (op[0] == 'A' && op[1] == 'N' && op[2] == 'D')
      instructions[i].op = AND;
    else if (op[0] == 'X' && op[1] == 'O' && op[2] == 'R')
      instructions[i].op = XOR;
  }
  fclose(file);

  int changed;
  do {
    changed = 0;
    for (int i = 0; i < M; i++) {
      Instruction ins = instructions[i];
      if (vals[ins.in1] == -1 || vals[ins.in2] == -1 || vals[ins.out] != -1) continue;

      changed++;
      switch (ins.op) {
        case OR:
          vals[ins.out] = vals[ins.in1] | vals[ins.in2];
          break;
        case AND:
          vals[ins.out] = vals[ins.in1] & vals[ins.in2];
          break;
        case XOR:
          vals[ins.out] = vals[ins.in1] ^ vals[ins.in2];
          break;
      }
    }
  } while (changed > 0);

  llu z = 0;
  for (int i = h("z99"); i >= h("z00"); i--) {
    if (vals[i] == -1) continue;
    z *= 2;
    z += vals[i];
  }

  printf("%llu\n", z);

  free(vals);

  return 0;
}
