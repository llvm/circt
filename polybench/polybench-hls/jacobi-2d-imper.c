#define PB_TSTEPS 4
#define PB_N 8

void kernel_jacobi_2d_imper(int tsteps, int n, int A[PB_N][PB_N],
                            B[PB_N][PB_N]) {
  int t, i, j;

  for (t = 0; t < PB_TSTEPS; t++) {
    for (i = 1; i < PB_N - 1; i++)
      for (j = 1; j < PB_N - 1; j++)
        B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] +
                         A[i - 1][j]);
    for (i = 1; i < PB_N - 1; i++)
      for (j = 1; j < PB_N - 1; j++)
        A[i][j] = B[i][j];
  }
}
