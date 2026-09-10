// RUN: hls-est %s --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis | FileCheck %s -check-prefix=CHECK-BRAM

// CHECK-BRAM: Total BRAM: 13

module {
  func.func @atax(%arg0: memref<64x64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "tmp1"} : memref<64xf32>
    %alloca_0 = memref.alloca() {hls.preserve, polygeist.varname = "buff_y_out"} : memref<64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "buff_x"} : memref<64xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "buff_A"} : memref<64x64xf32>
    affine.for %arg3 = 0 to 64 {
      %0 = affine.load %arg1[%arg3] : memref<64xf32>
      affine.store %0, %alloca_1[%arg3] : memref<64xf32>
      affine.store %cst, %alloca_0[%arg3] : memref<64xf32>
      affine.store %cst, %alloca[%arg3] : memref<64xf32>
      affine.for %arg4 = 0 to 64 {
        %1 = affine.load %arg0[%arg3, %arg4] : memref<64x64xf32>
        affine.store %1, %alloca_2[%arg3, %arg4] : memref<64x64xf32>
      }
    }
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        %0 = affine.load %alloca[%arg3] : memref<64xf32>
        %1 = affine.load %alloca_2[%arg3, %arg4] : memref<64x64xf32>
        %2 = affine.load %alloca_1[%arg4] : memref<64xf32>
        %3 = arith.mulf %1, %2 : f32
        %4 = arith.addf %0, %3 : f32
        affine.store %4, %alloca[%arg3] : memref<64xf32>
      }
    }
    affine.for %arg3 = 0 to 64 {
      %0 = affine.load %alloca[%arg3] : memref<64xf32>
      affine.for %arg4 = 0 to 64 {
        %1 = affine.load %alloca_0[%arg4] : memref<64xf32>
        %2 = affine.load %alloca_2[%arg3, %arg4] : memref<64x64xf32>
        %3 = arith.mulf %2, %0 : f32
        %4 = arith.addf %1, %3 : f32
        affine.store %4, %alloca_0[%arg4] : memref<64xf32>
      }
    }
    affine.for %arg3 = 0 to 64 {
      %0 = affine.load %alloca_0[%arg3] : memref<64xf32>
      affine.store %0, %arg2[%arg3] : memref<64xf32>
    }
    return
  }
}

// Kernel: atax
// 
// #define N 64
// 
// #define DATA_TYPE float
// #define SCALAR_VAL(x) x##f
// #define SQRT_FUN(x) sqrtf(x)
// #define EXP_FUN(x) expf(x)
// #define POW_FUN(x,y) powf(x,y)
// 
// void atax(DATA_TYPE A[N][N], DATA_TYPE x[N], DATA_TYPE y_out[N])
// {
//     int i, j;
//     DATA_TYPE buff_A[N][N];
//     DATA_TYPE buff_x[N];
//     DATA_TYPE buff_y_out[N];
//     DATA_TYPE tmp1[N];
// 
//     lprd_1: for (i = 0; i < N; i++) {
//         buff_x[i] = x[i];
//     	buff_y_out[i] = 0;
//         tmp1[i] = 0;
//     	lprd_2: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
//     		buff_A[i][j] = A[i][j];
//     	}
//     }
// 
//     lp1: for (i = 0; i < N; i++) {
//         lp2: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 	        tmp1[i] = tmp1[i] + buff_A[i][j] * buff_x[j];
//         }
//     }
// 
//     lp3: for (i = 0; i < N; i++) {
//         lp4: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
//         	buff_y_out[j] = buff_y_out[j] + buff_A[i][j] * tmp1[i];
//         }
//     }
// 
//     lpwr_1: for (i = 0; i < N; i++) {
//         y_out[i] = buff_y_out[i];
//     }
// }
// Target:
// +--------------+----------------------+---------+---+----+-----+------+-----+------+-------------+
// |    Memory    |        Module        | BRAM_18K| FF| LUT| URAM| Words| Bits| Banks| W*Bits*Banks|
// +--------------+----------------------+---------+---+----+-----+------+-----+------+-------------+
// |buff_A_U      |buff_A_RAM_AUTO_1R1W  |        8|  0|   0|    0|  4096|   32|     1|       131072|
// |buff_x_U      |buff_x_RAM_AUTO_1R1W  |        2|  0|   0|    0|    64|   32|     1|         2048|
// |buff_y_out_U  |buff_x_RAM_AUTO_1R1W  |        2|  0|   0|    0|    64|   32|     1|         2048|
// |tmp1_U        |tmp1_RAM_AUTO_1R1W    |        1|  0|   0|    0|    64|   32|     1|         2048|
// +--------------+----------------------+---------+---+----+-----+------+-----+------+-------------+
// |Total         |                      |       13|  0|   0|    0|  4288|  128|     4|       137216|
// +--------------+----------------------+---------+---+----+-----+------+-----+------+-------------+
