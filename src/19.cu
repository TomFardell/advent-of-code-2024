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

// Number of unmakeable patterns to store. Setting this too large slows the program, since there
// will be redundant values stored. I am using an array rather than a hash table, so the lookup time
// is proportional to this value
#define UC 20
#define MC 80        // Number of makeable patterns to store with their number of ways to make
#define LT (TL - 2)  // Longest towel is 2 less than the size of buffer needed

typedef long long unsigned llu;

typedef struct {
  char patterns[UC * PL];
  int n, p;
} unmakeables;

typedef struct {
  char patterns[MC * PL];
  llu counts[MC];
  int n, p;
} makeables;

llu ways_to_make(const char *pattern, const char *towels, unmakeables *u_store,
                 makeables *m_store) {
  int p_len = strlen(pattern);
  if (p_len == 0) return 1;

  llu ways = 0;

  // If pattern is memoized, return the count here
  for (int i = 0; i < m_store->n; i++) {
    if (!strcmp(pattern, m_store->patterns + PL * i)) {
      return m_store->counts[i];
    }
  }

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
        int unmakeable = 0;

        // The unmakeable patterns array is likely to contain null strings, so first ensure the
        // suffix is not null (as otherwise it will be seen as unmakeable)
        if (strlen(suffix) > 0) {
          for (int u = 0; u < u_store->n; u++) {
            if (!strcmp(suffix, u_store->patterns + PL * u)) {
              unmakeable = 1;
              break;
            }
          }
        }

        // If the suffix is stored as unmakeable, finish for this prefix
        if (unmakeable) break;

        llu suffix_ways = ways_to_make(suffix, towels, u_store, m_store);

        if (suffix_ways == 0) {
          // Put the suffix in the array of unmakeables, wrapping the count if the array is full
          u_store->p = (u_store->p + 1) % UC;
          u_store->n = (u_store->n + 1 > UC) ? UC : u_store->n + 1;
          strncpy(u_store->patterns + PL * u_store->p, suffix, PL);
        }

        ways += suffix_ways;
        break;  // Stop checking this prefix
      }
    }
  }

  // Add this pattern's data to the makeables storage
  m_store->p = (m_store->p + 1) % MC;
  m_store->n = (m_store->n + 1 > MC) ? MC : m_store->n + 1;
  strncpy(m_store->patterns + PL * m_store->p, pattern, PL);
  m_store->counts[m_store->p] = ways;

  return ways;
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

  unmakeables u_store = {{0}, 0, 0};
  makeables m_store = {{0}, {0}, 0, 0};

  int m_count = 0;
  llu t_count = 0;
  for (int p = 0; p < PC; p++) {
    llu w = ways_to_make(patterns + PL * p, towels, &u_store, &m_store);
    m_count += (w > 0);
    t_count += w;
  }

  printf("%d\n%llu\n", m_count, t_count);

  return 0;
}
