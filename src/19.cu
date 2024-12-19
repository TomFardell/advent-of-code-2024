#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define P

#ifdef P
#define FILE_NAME "data/19p.txt"
#define TC 447
#define PC 400
#define TL 10
#define PL 62
#endif
#ifdef S
#define FILE_NAME "data/19s.txt"
#define TC 8
#define PC 8
#define TL 5
#define PL 8
#endif

#define LT (TL - 2)     // Longest towel is 2 less than the size of buffer needed
#define MEMO_N 5        // Largest length of pattern to store in memo
#define MEMO_SIZE 7776  // 6 ^ MEMO_N

int char_coef(const char c) {
  switch (c) {
    case 'w':
      return 1;
    case 'u':
      return 2;
    case 'b':
      return 3;
    case 'r':
      return 4;
    case 'g':
      return 5;
    default:
      return 0;
  }
}

int hash(const char *pattern) {
  int res = 0;
  for (int i = 0; i < MEMO_N; i++) {
    int coef = char_coef(pattern[i]);
    // Return res as soon as the null terminator is found as there could be junk values after it
    if (!coef) return res;
    res *= MEMO_N + 1;
    res += coef;
  }
  return res;
}

int _can_be_made(const char *pattern, const char *used, const int n_used, const char *towels,
                 char *memo) {
  int p_len = strlen(pattern);
  int h = -1;
  if (p_len == 0) return 1;

  if (p_len <= MEMO_N) {
    h = hash(pattern);
    if (memo[h] != -1) {
      return memo[h];
    }
  }

  // For each size up to the minimum of this pattern's length and the maximum towel length
  for (int i = 0; i < ((p_len < LT) ? p_len : LT); i++) {
    // Get the prefix of pattern of this length
    char start[TL];
    strncpy(start, pattern, i + 1);
    start[i + 1] = 0;

    for (int t = 0; t < TC; t++) {
      // If the prefix matches a towel
      if (!strcmp(towels + TL * t, start)) {
        int not_used = 1;
        // Go through the used towels and check equality with the prefix
        for (int ut = 0; ut < n_used; ut++) {
          if (!strcmp(towels + TL * t, used + TL * ut)) {
            not_used = 0;
            break;
          }
        }

        // If the prefix is a valid unused towel, add that towel to a new used array and recurse
        if (not_used) {
          char *new_used = (char *)malloc((n_used + 1) * TL * sizeof(char));
          for (int ut = 0; ut < n_used; ut++) {
            strcpy(new_used + TL * ut, used + TL * ut);
          }
          strcpy(new_used + TL * n_used, start);

          int makeable = _can_be_made(pattern + i + 1, NULL, 0, towels, memo);
          free(new_used);
          if (makeable) {
            if (p_len <= MEMO_N) memo[h] = 1;
            return 1;
          }
        }
      }
    }
  }

  if (p_len <= MEMO_N) memo[h] = 0;
  return 0;
}

int can_be_made(const char *pattern, const char *towels) {
  char memo[MEMO_SIZE];
  for (int i = 0; i < MEMO_SIZE; i++) {
    memo[i] = -1;
  }

  return _can_be_made(pattern, NULL, 0, towels, memo);
}

int main() {
  char towels[TC * TL];
  char patterns[PC * PL];

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < TC - 1; i++) {
    fscanf(file, "%[^,]s", towels + TL * i);
    fscanf(file, ", ");
  }
  fscanf(file, "%s", towels + TL * (TC - 1));
  fscanf(file, "\n\n");
  for (int i = 0; i < PC; i++) {
    fscanf(file, "%s ", patterns + PL * i);
  }
  fclose(file);

  int count = 0;
  for (int p = 0; p < PC; p++) {
    count += can_be_made(patterns + PL * p, towels);
  }

  printf("%d\n", count);

  return 0;
}
