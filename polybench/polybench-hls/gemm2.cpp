// git.status = clean, build.date = Tue Apr 13 11:36:22 IST 2021, git.hash = fd7034e
#include <ap_int.h>
extern "C" {
  void kernel(ap_uint<32> C_int[2][2], ap_uint<32> A_int[2][2], ap_uint<32> B_int[2][2]) {
    #pragma HLS INTERFACE m_axi port=C_int offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=C_int bundle=control
    #pragma HLS INTERFACE m_axi port=A_int offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=A_int bundle=control
    #pragma HLS INTERFACE m_axi port=B_int offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=B_int bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    ap_uint<32> C[2][2];
    #pragma HLS resource variable=C core=RAM_1P_BRAM
    #pragma HLS ARRAY_PARTITION variable=C cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=C cyclic factor=2 dim=2
    ap_uint<32> A[2][2];
    #pragma HLS resource variable=A core=RAM_1P_BRAM
    #pragma HLS ARRAY_PARTITION variable=A cyclic factor=2 dim=1
    ap_uint<32> B[2][2];
    #pragma HLS resource variable=B core=RAM_1P_BRAM
    #pragma HLS ARRAY_PARTITION variable=B cyclic factor=2 dim=2
    for(int i = 0; i < 2; i++) {
      #pragma HLS UNROLL factor=1 skip_exit_check
      #pragma HLS LOOP_FLATTEN off
      for(int j = 0; j < 2; j++) {
        #pragma HLS UNROLL factor=1 skip_exit_check
        #pragma HLS LOOP_FLATTEN off
        A[i][j] = A_int[i][j];
        B[i][j] = B_int[i][j];
        C[i][j] = C_int[i][j];
      }
    }
    //---
    for(int i = 0; i < 2; i++) {
      #pragma HLS UNROLL factor=2 skip_exit_check
      #pragma HLS LOOP_FLATTEN off
      for(int j = 0; j < 2; j++) {
        #pragma HLS UNROLL factor=2 skip_exit_check
        #pragma HLS LOOP_FLATTEN off
        for(int k = 0; k < 2; k++) {
          #pragma HLS UNROLL factor=1 skip_exit_check
          #pragma HLS LOOP_FLATTEN off
          ap_uint<32> v = (A[i][k] * B[k][j]);
          // combiner:
          C[i][j] += v;
        }
      }
    }
    //---
    for(int i = 0; i < 2; i++) {
      #pragma HLS UNROLL factor=1 skip_exit_check
      #pragma HLS LOOP_FLATTEN off
      for(int j = 0; j < 2; j++) {
        #pragma HLS UNROLL factor=1 skip_exit_check
        #pragma HLS LOOP_FLATTEN off
        C_int[i][j] = C[i][j];
      }
    }
  }
}
