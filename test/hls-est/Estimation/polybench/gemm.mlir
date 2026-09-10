// RUN: hls-est %s  --auto-pipeline-unroll --canonicalize --affine-scalrep --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis | FileCheck %s -check-prefix=CHECK-BRAM

// CHECK-BRAM: Total BRAM: 32

module {
  func.func @gemm(%arg0: f32, %arg1: f32, %arg2: memref<64x64xf32>, %arg3: memref<64x64xf32>, %arg4: memref<64x64xf32>, %arg5: memref<64x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "tmp1"} : memref<64x64xf32>
    %alloca_0 = memref.alloca() {hls.preserve, polygeist.varname = "buff_C"} : memref<64x64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "buff_B"} : memref<64x64xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "buff_A"} : memref<64x64xf32>
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 64 {
        %0 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
        affine.store %0, %alloca_2[%arg6, %arg7] : memref<64x64xf32>
        %1 = affine.load %arg3[%arg6, %arg7] : memref<64x64xf32>
        affine.store %1, %alloca_1[%arg6, %arg7] : memref<64x64xf32>
        %2 = affine.load %arg4[%arg6, %arg7] : memref<64x64xf32>
        affine.store %2, %alloca_0[%arg6, %arg7] : memref<64x64xf32>
        affine.store %cst, %alloca[%arg6, %arg7] : memref<64x64xf32>
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 64 {
        affine.for %arg8 = 0 to 64 {
          %0 = affine.load %alloca_2[%arg6, %arg8] : memref<64x64xf32>
          %1 = arith.mulf %arg0, %0 : f32
          %2 = affine.load %alloca_1[%arg8, %arg7] : memref<64x64xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = affine.load %alloca[%arg6, %arg7] : memref<64x64xf32>
          %5 = arith.addf %3, %4 : f32
          affine.store %5, %alloca[%arg6, %arg7] : memref<64x64xf32>
        }
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 64 {
        %0 = affine.load %alloca_0[%arg6, %arg7] : memref<64x64xf32>
        %1 = arith.mulf %arg1, %0 : f32
        %2 = affine.load %alloca[%arg6, %arg7] : memref<64x64xf32>
        %3 = arith.addf %1, %2 : f32
        affine.store %3, %alloca_0[%arg6, %arg7] : memref<64x64xf32>
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 64 {
        %0 = affine.load %alloca_0[%arg6, %arg7] : memref<64x64xf32>
        affine.store %0, %arg5[%arg6, %arg7] : memref<64x64xf32>
      }
    }
    return
  }
}

// Kernel: gemm
// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
// #include <math.h>
// 
// #define N  64
// #define DATA_TYPE float
// #define SCALAR_VAL(x) x##f
// #define SQRT_FUN(x) sqrtf(x)
// #define EXP_FUN(x) expf(x)
// #define POW_FUN(x,y) powf(x,y)
// 
// void gemm(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D_out[N][N])
// {
//     int i, j, k;
// 
//     DATA_TYPE buff_A[N][N]; 
//     DATA_TYPE buff_B[N][N]; 
//     DATA_TYPE buff_C[N][N];
//     DATA_TYPE tmp1[N][N];
// 
//     lprd_1: for (i = 0; i < N; i++){
//         lprd_2: for (j = 0; j < N; j++){
//#pragma HLS unroll factor=1
//             buff_A[i][j] = A[i][j];
//             buff_B[i][j] = B[i][j];
//             buff_C[i][j] = C[i][j];
//             tmp1[i][j] = 0;
//         }
//     }
// 
//     lp1: for (i = 0; i < N; i++) {
//         lp2: for (j = 0; j < N; j++) {
//             lp3: for (k = 0; k < N; k++) {
//#pragma HLS unroll factor=1
//                 tmp1[i][j] = alpha * buff_A[i][k] * buff_B[k][j] + tmp1[i][j];
//             }
//         }
//     }
// 
//     lp4: for (i = 0; i < N; i++) {
//         lp5: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 	        buff_C[i][j] = beta * buff_C[i][j] + tmp1[i][j];
//         }
//     }
// 
//     lpwr_1: for (i = 0; i < N; i++){
//         lpwr_2: for (j = 0; j < N; j++){
//#pragma HLS unroll factor=1
//             D_out[i][j] = buff_C[i][j];
//         }
//     }
// }