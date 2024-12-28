#include <stdio.h>

#define P

#ifdef P
#define FILE_NAME "data/22p.txt"
#define N 1557
#endif
#ifdef S
#define FILE_NAME "data/22s.txt"
#define N 4
#endif

#define MOD 16777216
#define N_NUMS 2000

typedef long long unsigned llu;

llu evolve(llu n) {
  n = (n ^ (n * 64)) % MOD;
  n = n ^ (n / 32);
  n = (n ^ (n * 2048)) % MOD;

  return n;
}

int seq_value(const int *seq, llu n) {
  int diffs[4] = {10, 10, 10, 10};
  int prev = n % 10;
  for (int i = 0; i < N_NUMS; i++) {
    n = evolve(n);
    int price = n % 10;
    diffs[0] = diffs[1];
    diffs[1] = diffs[2];
    diffs[2] = diffs[3];
    diffs[3] = price - prev;
    prev = price;

    if (diffs[0] == seq[0] && diffs[1] == seq[1] && diffs[2] == seq[2] && diffs[3] == seq[3])
      return price;
  }

  return 0;
}

int seq_total_value(const int *seq, const llu *nums) {
  int total = 0;

  for (int i = 0; i < N; i++) {
    total += seq_value(seq, nums[i]);
  }

  return total;
}

int infeasible(const int n) { return (n < -9 || 9 < n); }

int main(void) {
  llu nums[N];

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < N; i++) {
    fscanf(file, "%llu", nums + i);
    fscanf(file, "\n");
  }
  fclose(file);

  llu total;
  for (int i = 0; i < N; i++) {
    llu n = nums[i];
    for (int j = 0; j < N_NUMS; j++) {
      n = evolve(n);
    }
    total += n;
  }

  int max_profit = 0;
  for (int a1 = -9; a1 <= 9; a1++) {
    for (int a2 = -9; a2 <= 9; a2++) {
      for (int a3 = -9; a3 <= 9; a3++) {
        for (int a4 = -9; a4 <= 9; a4++) {
          if (infeasible(a1 + a2 + a3 + a4) || infeasible(a1 + a2 + a3) ||
              infeasible(a2 + a3 + a4) || infeasible(a1 + a2) || infeasible(a2 + a3) ||
              infeasible(a3 + a4))
            continue;

          // Don't check this sequence if it isn't close in number of occurences to current best
          int seq[4] = {a1, a2, a3, a4};

          int val = seq_total_value(seq, nums);

          if (val > max_profit) max_profit = val;
          printf("%3d %3d %3d %3d: %6d (%6d)\n", a1, a2, a3, a4, val, max_profit);
        }
      }
    }
  }

  printf("%llu\n%d\n", total, max_profit);

  return 0;
}
