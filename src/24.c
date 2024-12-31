#include <stdio.h>
#include <stdlib.h>

#define P
#define PART2 1

#ifdef P
#define FILE_NAME "data/24p.txt"
#define N 90
#define M 222
#define X_SIZE 45
#define NUM_TESTS 50
#endif
#ifdef S  // Don't use part 2 outputs with the small input
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

// Numbers to use in regression testing
const llu test_nums[2 * NUM_TESTS] = {
    2883253254261,  10372030208467, 1542591425961,  12857922529081, 23133232767758, 24974970070797,
    9901518230099,  4884676053340,  11560743780238, 13104048093,    27712403137476, 5662885895767,
    8715261008592,  21001585684710, 13518205375431, 2121631666479,  16012451976662, 34141917057322,
    6970284611951,  7441143543581,  3072009209825,  33094936216960, 32759201667043, 12107610526494,
    13717389102996, 22318861119945, 23918373018660, 23715782916009, 18473611378694, 9284926886308,
    31190850431641, 26212557961358, 32918445961290, 12946326078503, 32504071345982, 31038721405236,
    2250695622594,  28567575646161, 33236733591712, 10101690247423, 15333353862432, 25780403469138,
    32270294245013, 15458485415463, 17007372115710, 20459658506675, 22522928926417, 26641358049952,
    12309285002055, 6286696464017,  23654425971872, 19605845039761, 19956460104979, 18507848049681,
    17525265548412, 27153403035096, 10861145429491, 33267837035177, 3754420224103,  30198011561690,
    11711590407163, 27153891437651, 23292194966158, 34109713045208, 3671533691924,  12061737485902,
    8608499472346,  6645339563028,  5557640138808,  32957967394956, 29153325111811, 14386933339973,
    1093956865423,  18257439555430, 29104320586594, 19263249740643, 20675144039321, 29741922517781,
    24666126339866, 3661155081708,  5212319414551,  11946451970488, 25127923503937, 6436465081120,
    19599113251174, 12934369236655, 23953697822461, 23467056233015, 11511687772487, 24067077630680,
    10914611234022, 25189725134786, 29378890578039, 2375209043694,  28972150757838, 28938209947033,
    29915917985,    29201980399713, 19731854013858, 7732174046798};

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

void print_binary(const llu x, const int n_bits) {
  for (llu si = 1LLU << (n_bits - 1); si > 0; si >>= 1) {
    printf("%d", (x & si) != 0);
  }
}

void print_instruction(Instruction ins) {
  char in1[4], in2[4], out[4];
  h_inv(ins.in1, in1);
  h_inv(ins.in2, in2);
  h_inv(ins.out, out);

  printf("%s ", in1);

  switch (ins.op) {
    case OR:
      printf("OR  ");
      break;
    case AND:
      printf("AND ");
      break;
    case XOR:
      printf("XOR ");
      break;
  }

  printf("%s -> %s\n", in2, out);
}

void print_dependence(const int n_layers, const int out, const Instruction *instructions) {
  if (n_layers == 0) return;

  for (int i = 0; i < M; i++) {
    if (instructions[i].out != out) continue;

    print_dependence(n_layers - 1, instructions[i].in1, instructions);
    print_dependence(n_layers - 1, instructions[i].in2, instructions);
    print_instruction(instructions[i]);
  }
}

llu simulate_program(const llu x, const llu y, const Instruction *instructions) {
  int *vals = malloc((sizeof *vals) * A * A * A);
  for (int i = 0; i < A * A * A; i++) {
    vals[i] = -1;
  }

  llu b = 1;
  for (int i = 0; i < X_SIZE; i++, b <<= 1) {
    char xseq[4], yseq[4];
    sprintf(xseq, "x%02d", i);
    sprintf(yseq, "y%02d", i);
    vals[h(xseq)] = (x & b) != 0;
    vals[h(yseq)] = (y & b) != 0;
  }

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

  free(vals);

  return z;
}

double err(const Instruction *instructions) {
  double error_count = 0;

  for (int i = 0; i < NUM_TESTS; i++) {
    llu x = test_nums[2 * i];
    llu y = test_nums[2 * i + 1];
    llu z = simulate_program(x, y, instructions);

    llu b = 1;
    for (int i = 0; i < X_SIZE + 1; i++, b <<= 1) {
      int xybit = ((x + y) & b) != 0;
      int zbit = (z & b) != 0;
      error_count += xybit != zbit;
    }
  }

  return error_count / (X_SIZE * NUM_TESTS);
}

int comp_int(const void *a, const void *b) { return *((int *)a) - *((int *)b); }

int main(void) {
  llu x = 0, y = 0;
  Instruction instructions[M];

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < X_SIZE; i++) {
    char dummy[4];
    llu bit;
    fscanf(file, "%3s", dummy);
    fscanf(file, ": %llu ", &bit);
    x += bit << i;
  }
  for (int i = 0; i < X_SIZE; i++) {
    char dummy[4];
    llu bit;
    fscanf(file, "%3s", dummy);
    fscanf(file, ": %llu ", &bit);
    y += bit << i;
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

#if PART2 == 1
  int swaps[8];

  for (int pair = 0; pair < 4; pair++) {
    double min_error = err(instructions);
    int m1 = -1, m2 = -1;  // Instructions whose swap yields the minimum error

    for (int i1 = 0; i1 < M; i1++) {
      for (int i2 = i1 + 1; i2 < M; i2++) {
        // Swap the two instruction outputs
        int temp = instructions[i1].out;
        instructions[i1].out = instructions[i2].out;
        instructions[i2].out = temp;

        // Calculate the error with these instructions swapped
        double error = err(instructions);

        if (error < min_error) {
          printf("%d %d %lf\n", instructions[i1].out, instructions[i2].out, error);
          min_error = error;
          m1 = i1;
          m2 = i2;
        }

        // Swap the instruction outputs back
        temp = instructions[i1].out;
        instructions[i1].out = instructions[i2].out;
        instructions[i2].out = temp;
      }
    }

    // Permanently swap the minimising instructions' outputs
    int temp = instructions[m1].out;
    instructions[m1].out = instructions[m2].out;
    instructions[m2].out = temp;

    swaps[2 * pair] = instructions[m1].out;
    swaps[2 * pair + 1] = instructions[m2].out;
  }

  qsort(swaps, 8, sizeof(int), comp_int);
  for (int i = 0; i < 7; i++) {
    char seq[4];
    h_inv(swaps[i], seq);
    printf("%s,", seq);
  }
  char seq[4];
  h_inv(swaps[7], seq);
  printf("%s\n", seq);
#else

  llu z = simulate_program(x, y, instructions);
  printf("%llu\n", z);
#endif

  return 0;
}
