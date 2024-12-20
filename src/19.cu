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

// Number of makeable and unmakeable patterns to store. Setting this too large slows the program,
// since there will be redundant values stored. I am using an array rather than a hash table, so the
// lookup time is proportional to this value
#define MC 100
#define LT (TL - 2)  // Longest towel is 2 less than the size of buffer needed

typedef struct {
  char makeables[MC * PL], unmakeables[MC * PL];
  int n_m, n_u;
  int c_m, c_u;
} storage;

int can_be_made(const char *pattern, const char *towels, storage *p_store) {
  int p_len = strlen(pattern);
  if (p_len == 0) return 1;

  // For each size up to the minimum of this pattern's length and the maximum towel length
  for (int i = 0; i < ((p_len < LT) ? p_len : LT); i++) {
    // Get the prefix of pattern of this length
    char prefix[TL];
    strncpy(prefix, pattern, i + 1);
    prefix[i + 1] = 0;

    for (int t = 0; t < TC; t++) {
      // If the prefix matches a towel
      if (!strcmp(towels + TL * t, prefix)) {
        const char *suffix = pattern + i + 1;
        int makeable = -1;
        int new_suffix = 1;

        for (int m = 0; m < p_store->n_m && makeable == -1; m++) {
          if (!strcmp(suffix, p_store->makeables + PL * m)) {
            makeable = 1;
          }
        }

        // p_store->unmakeables is likely to contain null strings, so first ensure the suffix is not
        // null (as otherwise it will be seen as unmakeable)
        if (strlen(suffix) > 0) {
          for (int u = 0; u < p_store->n_u && makeable == -1; u++) {
            if (!strcmp(suffix, p_store->unmakeables + PL * u)) {
              makeable = 0;
              new_suffix = 0;
            }
          }
        }

        // Only recurse if not already seen the suffix
        if (makeable == -1) makeable = can_be_made(suffix, towels, p_store);

        if (makeable) {
          // Put the pattern in the array of makeables, wrapping the count if the array is full
          p_store->c_m = (p_store->c_m + 1) % MC;
          p_store->n_m = (p_store->n_m + 1 > MC) ? MC : p_store->n_m + 1;
          strncpy(p_store->makeables + PL * p_store->c_m, pattern, PL);
          return 1;
        } else if (new_suffix) {
          // Put the suffix in the array of unmakeables, wrapping the count if the array is full
          p_store->c_u = (p_store->c_u + 1) % MC;
          p_store->n_u = (p_store->n_u + 1 > MC) ? MC : p_store->n_u + 1;
          strncpy(p_store->unmakeables + PL * p_store->c_u, suffix, PL);
        }
      }
    }
  }

  return 0;
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

  storage p_store = {{0}, {0}, 0, 0, 0, 0};

  int count = 0;
  for (int p = 0; p < PC; p++) {
    int m = can_be_made(patterns + PL * p, towels, &p_store);
    count += m;
  }

  printf("%d\n", count);

  return 0;
}
