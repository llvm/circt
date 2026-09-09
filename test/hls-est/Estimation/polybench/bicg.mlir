// RUN: hls-est %s --auto-pipeline-unroll --canonicalize --affine-scalrep --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis | FileCheck %s -check-prefix=CHECK-BRAM

// CHECK-BRAM: Total BRAM: 13

module {
  func.func @bicg(%arg0: memref<64x64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "buff_q_out"} : memref<64xf32>
    %alloca_0 = memref.alloca() {hls.preserve, polygeist.varname = "buff_s_out"} : memref<64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "buff_r"} : memref<64xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "buff_p"} : memref<64xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "buff_A"} : memref<64x64xf32>
    affine.for %arg5 = 0 to 64 {
      %0 = affine.load %arg1[%arg5] : memref<64xf32>
      affine.store %0, %alloca_2[%arg5] : memref<64xf32>
      %1 = affine.load %arg2[%arg5] : memref<64xf32>
      affine.store %1, %alloca_1[%arg5] : memref<64xf32>
      affine.store %cst, %alloca_0[%arg5] : memref<64xf32>
      affine.store %cst, %alloca[%arg5] : memref<64xf32>
      affine.for %arg6 = 0 to 64 {
        %2 = affine.load %arg0[%arg5, %arg6] : memref<64x64xf32>
        affine.store %2, %alloca_3[%arg5, %arg6] : memref<64x64xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      %0 = affine.load %alloca_1[%arg5] : memref<64xf32>
      affine.for %arg6 = 0 to 64 {
        %1 = affine.load %alloca_0[%arg6] : memref<64xf32>
        %2 = affine.load %alloca_3[%arg5, %arg6] : memref<64x64xf32>
        %3 = arith.mulf %2, %0 : f32
        %4 = arith.addf %1, %3 : f32
        affine.store %4, %alloca_0[%arg6] : memref<64xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      %0 = affine.load %alloca[%arg5] : memref<64xf32>
      %1 = affine.for %arg6 = 0 to 64 iter_args(%acc = %0) -> (f32) {
        %2 = affine.load %alloca_3[%arg5, %arg6] : memref<64x64xf32>
        %3 = affine.load %alloca_2[%arg6] : memref<64xf32>
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %acc, %4 : f32
        affine.yield %5 : f32
      }
      affine.store %1, %alloca[%arg5] : memref<64xf32>
    }
    affine.for %arg5 = 0 to 64 {
      %0 = affine.load %alloca_0[%arg5] : memref<64xf32>
      affine.store %0, %arg3[%arg5] : memref<64xf32>
      %1 = affine.load %alloca[%arg5] : memref<64xf32>
      affine.store %1, %arg4[%arg5] : memref<64xf32>
    }
    return
  }
}

// Kernel: bicg
//
// #define N 64
// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
// #include <math.h>
// 
// #define DATA_TYPE float
// #define SCALAR_VAL(x) x##f
// #define SQRT_FUN(x) sqrtf(x)
// #define EXP_FUN(x) expf(x)
// #define POW_FUN(x,y) powf(x,y)
// 
// void bicg(DATA_TYPE A[N][N], DATA_TYPE p[N], DATA_TYPE r[N], DATA_TYPE s_out[N], DATA_TYPE q_out[N])
// {
//     int i, j;
// 
// 	DATA_TYPE buff_A[N][N];
// 	DATA_TYPE buff_p[N];
// 	DATA_TYPE buff_r[N];
// 	DATA_TYPE buff_s_out[N];
// 	DATA_TYPE buff_q_out[N];
// 
// 	lprd_1: for (i = 0; i < N; i++) {
// 		buff_p[i] = p[i];
// 		buff_r[i] = r[i];
// 		buff_s_out[i] = 0;
// 		buff_q_out[i] = 0;
// 		lprd_2: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 			buff_A[i][j] = A[i][j];
// 		}
// 	}
// 
//  lp1: for (i = 0; i < N; i++) {
//         lp2: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 	        buff_s_out[j] = buff_s_out[j] + buff_A[i][j] * buff_r[i];
// 		}
// 	}
// 
// 	lp3: for (i = 0; i < N; i++) {
//         lp4: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 	        buff_q_out[i] = buff_q_out[i] + buff_A[i][j] * buff_p[j];
// 	    }
//  }
// 
// 	lpwr: for (i = 0; i < N; i++) {
// 		s_out[i] = buff_s_out[i];
// 		q_out[i] = buff_q_out[i];
// 	}
// }
// +--------------+----------------------+---------+---+----+-----+------+-----+------+-------------+
// |    Memory    |        Module        | BRAM_18K| FF| LUT| URAM| Words| Bits| Banks| W*Bits*Banks|
// +--------------+----------------------+---------+---+----+-----+------+-----+------+-------------+
// |buff_A_U      |buff_A_RAM_AUTO_1R1W  |        8|  0|   0|    0|  4096|   32|     1|       131072|
// |buff_p_U      |buff_p_RAM_AUTO_1R1W  |        2|  0|   0|    0|    64|   32|     1|         2048|
// |buff_s_out_U  |buff_p_RAM_AUTO_1R1W  |        2|  0|   0|    0|    64|   32|     1|         2048|
// |buff_r_U      |buff_r_RAM_AUTO_1R1W  |        1|  0|   0|    0|    64|   32|     1|         2048|
// |buff_q_out_U  |buff_r_RAM_AUTO_1R1W  |        1|  0|   0|    0|    64|   32|     1|         2048|
// +--------------+----------------------+---------+---+----+-----+------+-----+------+-------------+
// |Total         |                      |       14|  0|   0|    0|  4352|  160|     5|       139264|
// +--------------+----------------------+---------+---+----+-----+------+-----+------+-------------+

