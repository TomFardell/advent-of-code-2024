#include <stdio.h>
#include <stdlib.h>

#define P

#ifdef P
#define FILE_NAME "data/23p.txt"
#define N 3380
#define MAX_CLIQUE 14
#define MAX_SEARCH 13
#endif
#ifdef S
#define FILE_NAME "data/23s.txt"
#define N 32
#define MAX_CLIQUE 5
#define MAX_SEARCH 4
#endif

#define A 26
#define A_SQ (A * A)

int h(const char *name) { return (name[1] - 'a') + A * (name[0] - 'a'); }

void h_inv(const int n, char *buffer) {
  buffer[0] = 'a' + (n / A);
  buffer[1] = 'a' + (n % A);
}

int comp_int(const void *a, const void *b) { return *((int *)a) - *((int *)b); }

// Return the largest clique containing the n input nodes, assuming they form a clique already.
// Since every node in the graph has degree 13, the largest possible clique would be of size 14, so
// nodes should be an array of length 14. Note the first n elements of nodes are constant, but nodes
// will contain the full largest clique. Stops searching if finds a clique of size MAX_SEARCH
int largest_clique(int *nodes, const int n, const int *connected) {
  int largest = n;

  // For each possible node
  for (int c_add = 0; c_add < A_SQ; c_add++) {
    int is_connected = 1;

    // Go through the current nodes and see if they are all connected to the candidate node
    for (int i = 0; i < n; i++) {
      if (!connected[nodes[i] + A_SQ * c_add]) {
        is_connected = 0;
        break;
      }
    }

    // If they are, add the candidate and recurse
    if (is_connected) {
      nodes[n] = c_add;
      int this_largest = largest_clique(nodes, n + 1, connected);
      if (this_largest > largest) {
        largest = this_largest;
        if (largest == MAX_SEARCH) break;
      }
    }
  }

  return largest;
}

int main(void) {
  int *connected = calloc(A_SQ * A_SQ, sizeof *connected);

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < N; i++) {
    char n1[2], n2[2];
    n1[0] = fgetc(file);
    n1[1] = fgetc(file);
    fgetc(file);  // Read hyphen
    n2[0] = fgetc(file);
    n2[1] = fgetc(file);
    fscanf(file, " ");  // Read newline
    connected[h(n1) + A_SQ * h(n2)] = 1;
    connected[h(n2) + A_SQ * h(n1)] = 1;
  }
  fclose(file);

  int count = 0;
  for (int c1 = 0; c1 < A_SQ; c1++) {
    for (int c2 = 0; c2 < A_SQ; c2++) {
      if (!connected[c1 + A_SQ * c2]) continue;

      for (int c3 = 0; c3 < A_SQ; c3++) {
        if (!connected[c2 + A_SQ * c3] || !connected[c3 + A_SQ * c1]) continue;

        char s1[2], s2[2], s3[2];
        h_inv(c1, s1);
        h_inv(c2, s2);
        h_inv(c3, s3);
        if (s1[0] == 't' || s2[0] == 't' || s3[0] == 't') count++;
      }
    }
  }

  // Every node is counted 6 times (once for each permuation of the party)
  count /= 6;

  printf("%d\n", count);

  int nodes[MAX_CLIQUE];
  int n = largest_clique(nodes, 0, connected);
  qsort(nodes, n, sizeof *nodes, comp_int);

  for (int i = 0; i < n - 1; i++) {
    char s[2];
    h_inv(nodes[i], s);
    printf("%c%c,", s[0], s[1]);
  }
  char s[2];
  h_inv(nodes[n - 1], s);
  printf("%c%c\n", s[0], s[1]);

  free(connected);

  return 0;
}
