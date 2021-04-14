#include <ap_int.h>
void gesummv_hls(ap_uint<32> alpha, ap_uint<32> beta, ap_uint<32> tmp[8],
                 ap_uint<32> A[8][8], ap_uint<32> B[8][8], ap_uint<32> x[8],
                 ap_uint<32> y[8]) {

  for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL factor = 1 skip_exit_check
#pragma HLS LOOP_FLATTEN off
    int tmp_reg = 0;
    int y_reg = 0;
    for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL factor = 1 skip_exit_check
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE
      ap_uint<32> t1 = (A[i][j] * x[j]);
      ap_uint<32> t2 = (B[i][j] * x[j]);
      // combiner:
      tmp_reg += t1;
      y_reg += t2;
    }
    tmp[i] = tmp_reg;
    //---
    ap_uint<32> y_i = y_reg;
    //---
    y[i] = ((alpha * tmp[i]) + (beta * y_i));
  }
}
