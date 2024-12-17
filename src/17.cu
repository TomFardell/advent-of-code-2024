//=====| NO PARALLELISM FOR THIS PUZZLE |===========================================================
// I have explained my implementation of part 2 of this puzzle in and around the _find_a method. The
// problems are starting to get quite difficult, so I will probably not use parallelism for any of
// the later ones, unless it would work especially well.
//==================================================================================================
#include <stdio.h>
#include <stdlib.h>

#define P

#ifdef P
#define FILE_NAME "data/17p.txt"
#define N 16
#endif
#ifdef S1
#define FILE_NAME "data/17s1.txt"
#define N 6
#endif
#ifdef S2
#define FILE_NAME "data/17s2.txt"
#define N 6
#endif

typedef long long ll;

typedef struct {
  int p;       // Program pointer after program step
  ll a, b, c;  // Value of registers after step
  int o;       // Output to print after this step (-1 for no output)
} program_state;

ll combo(const int operand, const ll a, const ll b, const ll c) {
  if (operand <= 3) return operand;
  if (operand == 4) return a;
  if (operand == 5) return b;
  if (operand == 6) return c;

  printf("Program is invalid\n");
  exit(EXIT_FAILURE);
}

program_state step_program(const program_state state, const int *program) {
  int p = state.p;
  ll a = state.a, b = state.b, c = state.c;
  int o = -1;

  int opcode = program[state.p];
  int operand = program[state.p + 1];
  p += 2;

  switch (opcode) {
    case 0: {
      for (int i = 0; i < combo(operand, a, b, c); i++) a /= 2;
      break;
    }
    case 1: {
      b = b ^ (ll)operand;
      break;
    }
    case 2: {
      b = combo(operand, a, b, c) % 8;
      break;
    }
    case 3: {
      if (a != 0) {
        p = operand;
      }
      break;
    }
    case 4: {
      b = b ^ c;
      break;
    }
    case 5: {
      o = combo(operand, a, b, c) % 8;
      break;
    }
    case 6: {
      ll d = a;
      for (int i = 0; i < combo(operand, a, b, c); i++) d /= 2;
      b = d;
      break;
    }
    case 7: {
      ll d = a;
      for (int i = 0; i < combo(operand, a, b, c); i++) d /= 2;
      c = d;
      break;
    }
  }

  return {p, a, b, c, o};
}

int output_after_loop(const ll a, const int *program) {
  program_state state = {0, a, 0, 0, -1};
  // Assuming program ends with a goto start
  while (state.p < N - 2) {
    state = step_program(state, program);
  }

  // Assuming the last instruction before the goto is an output
  return state.o;
}

// Splitting our valid A into: a = a_0 + 8 * a_1 + 8^2 * a_2 + ... + 8^(N-1) * a_(N-1), where the
// a_i are all between 0 and 7 (inclusive), for a given i, we have curr_a = a_i + 8 * a_(i+1) + 8^2
// * a_(i+2) + ... + 8^(N-1-i) * a_(N-1). Note that I am assuming the following about the program:
// - the program ends in a GOTO 0
// - there is one output per loop of the program. Let O = A, B or C be the output register
// - for each loop of the program, O is set entirely depending on A
// - for each loop of the program, A is divided by 8 and O does not use A after this division
ll *_find_a(const ll curr_a, const int i, const int *program) {
  // Base case, return a single element array containing this value of a
  if (i == 0) {
    ll *arr = (ll *)malloc(2 * sizeof(ll));
    arr[0] = curr_a;
    arr[1] = -1;
    return arr;
  }

  ll next_a[8];
  int p = 0;

  // Test the 8 values of a_(i-1). Keep the ones that give the correct program output in this loop
  for (int aim1 = 0; aim1 < 8; aim1++) {
    if (output_after_loop(8 * curr_a + aim1, program) == program[i - 1]) {
      next_a[p] = 8 * curr_a + aim1;
      p++;
    }
  }

  // arr holds the values of a which are valid at this level
  ll *arr = (ll *)malloc(sizeof(ll));
  int dp = 0;
  arr[dp++] = -1;

  // For each of our valid a_(i-1), recurse with the corresponding value of a
  for (int j = 0; j < p; j++) {
    ll a = next_a[j];
    ll *next_a = _find_a(a, i - 1, program);
    for (int k = 0; next_a[k] != -1; k++) {
      arr = (ll *)realloc(arr, ++dp * sizeof(ll));
      arr[dp - 2] = next_a[k];
      arr[dp - 1] = -1;
    }
    free(next_a);
  }

  return arr;
}

// Returns a dynamically allocated array of starting values for A that cause the program to output
// itself (terminating with -1). Wrapper for the recursive function
ll *find_a(const int *program) { return _find_a(0, N, program); }

int main(void) {
  int program[N];
  program_state state = {0, 0, 0, 0, -1};

  FILE *file = fopen(FILE_NAME, "r");
  fscanf(file, "Register A: %lld\n", &state.a);
  fscanf(file, "Register B: %lld\n", &state.b);
  fscanf(file, "Register C: %lld\n\nProgram: ", &state.c);
  for (int i = 0; i < N; i++) {
    fscanf(file, "%d", program + i);
    fscanf(file, ",");
  }
  fclose(file);

  // Part 1 program output
  while (state.p < N) {
    state = step_program(state, program);
    if (state.o != -1) {
      printf("%d,", state.o);
    }
  }
  printf("\n");

  ll *found_a = find_a(program);
  ll min_a = found_a[0];

  for (int i = 1; found_a[i] != -1; i++) {
    if (found_a[i] < min_a) min_a = found_a[i];
  }

  printf("%lld\n", min_a);

  free(found_a);

  return 0;
}
