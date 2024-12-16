# ADVENT OF CODE 2024 (CUDA)

My Advent of Code 2024 repository. I am using CUDA where GPU parallelism is appropriate. I will most likely not use many C++ features, so this repository also serves as a C implementation of the problems.

### Key for puzzle input types:
- **p** - personal puzzle input. These text files are not committed here, since personal puzzle inputs are not supposed to be shared.
- **s** - small puzzle input. This is the input given in each puzzle's description.
- **d** - debug puzzle input. These are custom inputs that I have set up to test certain edge cases.

### List of puzzles parallelised:
Not every puzzle suited parallelism, and there were some days where I simply did not want to program in CUDA (since the code for moving data between the host and GPU can be very verbose). The following puzzles call a CUDA kernel:

**[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]**
