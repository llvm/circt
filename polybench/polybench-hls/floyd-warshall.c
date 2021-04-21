void kernel_floyd_warshall(int n, int path[8][8]) {
#pragma HLS INTERFACE ap_ctrl_none port=return
  int i, j, k;

  for (k = 0; k < 8; k++) {
    for (i = 0; i < 8; i++)
      for (j = 0; j < 8; j++)
#pragma HLS PIPELINE
        path[i][j] = path[i][j] < path[i][k] + path[k][j]
                         ? path[i][j]
                         : path[i][k] + path[k][j];
  }
}
