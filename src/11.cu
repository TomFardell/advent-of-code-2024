//=====| NO PARALLELISM FOR THIS PUZZLE |==========================================================
// The recursion depth of 24 for device functions meant I couldn't run the recursive algorithm on
// the GPU. I tried implementing a stack based algorithm, but I couldn't figure out how to memoize
// this. I have committed the memoized recusrive algorithm with no kernel call. For part 1, I was
// able to solve with a kernel by first precomputing the specific stone numbers after some number of
// blinks (for example, after 12 blinks there were around 800 stones). I then launched a kernel to
// compute the number of resultant stones from each of these stones after the remaining blinks.
//=================================================================================================

#include <stdio.h>
#include <stdlib.h>

#define INPUT 'p'

#if INPUT == 'p'
#define FILE_NAME "data/11p.txt"
#define N 8
#elif INPUT == 's'
#define FILE_NAME "data/11s.txt"
#define N 2
#endif

#define NUM_BLINKS_P1 25
#define NUM_BLINKS_P2 75
#define MAX_MEMO_NUM 10  // Not inclusive. Memoize only small numbers to avoid hashing

typedef unsigned long long u64;

typedef struct {
  int count;  // Whether this is 1 or 2 elements (or 0 I guess)
  u64 nums[2];
} possible_pair;

int num_digits(const u64 n) {
  int d;
  u64 k = n;
  for (d = 0; k > 0; d++) {
    k /= 10;
  }

  return d;
}

u64 u64pow(const int n, const int base) {
  u64 r = 1;
  for (int i = 0; i < n; i++) {
    r *= base;
  }
  return r;
}

possible_pair get_next_stones(const u64 num) {
  possible_pair stones;
  if (num == 0) {
    stones.count = 1;
    stones.nums[0] = 1;
    return stones;
  }

  int n_digits = num_digits(num);
  if (n_digits % 2 == 0) {
    u64 divisor = u64pow(n_digits / 2, 10);
    stones.count = 2;
    stones.nums[0] = num / divisor;
    stones.nums[1] = num % divisor;
    return stones;
  }

  stones.count = 1;
  stones.nums[0] = num * 2024;
  return stones;
}

u64 num_resultant_stones(const u64 num, const int n_blinks, u64 *memo) {
  if (n_blinks == 0) return 1;

  // Use memoization for small values of num
  if (num < MAX_MEMO_NUM && (memo[num + MAX_MEMO_NUM * (n_blinks - 1)] != 0))
    return memo[num + MAX_MEMO_NUM * (n_blinks - 1)];

  u64 result;
  possible_pair next_stones = get_next_stones(num);
  if (next_stones.count == 1)
    result = num_resultant_stones(next_stones.nums[0], n_blinks - 1, memo);
  else
    result = num_resultant_stones(next_stones.nums[0], n_blinks - 1, memo) +
             num_resultant_stones(next_stones.nums[1], n_blinks - 1, memo);

  if (num < MAX_MEMO_NUM) memo[num + MAX_MEMO_NUM * (n_blinks - 1)] = result;
  return result;
}

int main(void) {
  u64 nums[N];

  FILE *file = fopen(FILE_NAME, "r");
  for (int i = 0; i < N; i++) {
    fscanf(file, "%llu ", nums + i);
  }
  fclose(file);

  u64 *memo = (u64 *)calloc(NUM_BLINKS_P2 * MAX_MEMO_NUM, sizeof(u64));
  u64 total1 = 0, total2 = 0;
  for (int i = 0; i < N; i++) {
    total1 += num_resultant_stones(nums[i], NUM_BLINKS_P1, memo);
    total2 += num_resultant_stones(nums[i], NUM_BLINKS_P2, memo);
  }
  free(memo);

  printf("%llu\n%llu\n", total1, total2);

  return 0;
}
