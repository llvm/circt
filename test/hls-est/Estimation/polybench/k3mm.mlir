// RUN: hls-est %s --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis 

// Total BRAM: 56

module {
  func.func @k3mm(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>, %arg3: memref<64x64xf32>, %arg4: memref<64x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "tmp2"} : memref<64x64xf32>
    %alloca_0 = memref.alloca() {hls.preserve, polygeist.varname = "tmp1"} : memref<64x64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "buff_E_out"} : memref<64x64xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "buff_D"} : memref<64x64xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "buff_C"} : memref<64x64xf32>
    %alloca_4 = memref.alloca() {hls.preserve, polygeist.varname = "buff_B"} : memref<64x64xf32>
    %alloca_5 = memref.alloca() {hls.preserve, polygeist.varname = "buff_A"} : memref<64x64xf32>
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %0 = affine.load %arg0[%arg5, %arg6] : memref<64x64xf32>
        affine.store %0, %alloca_5[%arg5, %arg6] : memref<64x64xf32>
        %1 = affine.load %arg1[%arg5, %arg6] : memref<64x64xf32>
        affine.store %1, %alloca_4[%arg5, %arg6] : memref<64x64xf32>
        %2 = affine.load %arg2[%arg5, %arg6] : memref<64x64xf32>
        affine.store %2, %alloca_3[%arg5, %arg6] : memref<64x64xf32>
        %3 = affine.load %arg3[%arg5, %arg6] : memref<64x64xf32>
        affine.store %3, %alloca_2[%arg5, %arg6] : memref<64x64xf32>
        affine.store %cst, %alloca_1[%arg5, %arg6] : memref<64x64xf32>
        affine.store %cst, %alloca_0[%arg5, %arg6] : memref<64x64xf32>
        affine.store %cst, %alloca[%arg5, %arg6] : memref<64x64xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %0 = affine.load %alloca_0[%arg5, %arg6] : memref<64x64xf32>
        %1 = affine.for %arg7 = 0 to 64 iter_args(%acc = %0) -> (f32) {
          %2 = affine.load %alloca_5[%arg5, %arg7] : memref<64x64xf32>
          %3 = affine.load %alloca_4[%arg7, %arg6] : memref<64x64xf32>
          %4 = arith.mulf %2, %3 : f32
          %5 = arith.addf %acc, %4 : f32
          affine.yield %5 : f32
        }
        affine.store %1, %alloca_0[%arg5, %arg6] : memref<64x64xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %0 = affine.load %alloca[%arg5, %arg6] : memref<64x64xf32>
        %1 = affine.for %arg7 = 0 to 64 iter_args(%acc = %0) -> (f32) {
          %2 = affine.load %alloca_3[%arg5, %arg7] : memref<64x64xf32>
          %3 = affine.load %alloca_2[%arg7, %arg6] : memref<64x64xf32>
          %4 = arith.mulf %2, %3 : f32
          %5 = arith.addf %acc, %4 : f32
          affine.yield %5 : f32
        }
        affine.store %1, %alloca[%arg5, %arg6] : memref<64x64xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %0 = affine.load %alloca_1[%arg5, %arg6] : memref<64x64xf32>
        %1 = affine.for %arg7 = 0 to 64 iter_args(%acc = %0) -> (f32) {
          %2 = affine.load %alloca_0[%arg5, %arg7] : memref<64x64xf32>
          %3 = affine.load %alloca[%arg7, %arg6] : memref<64x64xf32>
          %4 = arith.mulf %2, %3 : f32
          %5 = arith.addf %acc, %4 : f32
          affine.yield %5 : f32
        }
        affine.store %1, %alloca_1[%arg5, %arg6] : memref<64x64xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %0 = affine.load %alloca_1[%arg5, %arg6] : memref<64x64xf32>
        affine.store %0, %arg4[%arg5, %arg6] : memref<64x64xf32>
      }
    }
    return
  }
}

// Kernel: k3mm
// 
// #define N 64
// 
// #define DATA_TYPE float
// #define SCALAR_VAL(x) x##f
// #define SQRT_FUN(x) sqrtf(x)
// #define EXP_FUN(x) expf(x)
// #define POW_FUN(x,y) powf(x,y)
// 
// void k3mm(DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D[N][N], DATA_TYPE E_out[N][N])
// {
//     int i, j, k;
// 
//     DATA_TYPE buff_A[N][N];
//     DATA_TYPE buff_B[N][N];
//     DATA_TYPE buff_C[N][N];
//     DATA_TYPE buff_D[N][N];
//     DATA_TYPE buff_E_out[N][N];
//     DATA_TYPE tmp1[N][N];
//     DATA_TYPE tmp2[N][N];
// 
//     lprd_1: for (i = 0; i < N; i++) {
//         lprd_2: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
//             buff_A[i][j] = A[i][j];
//             buff_B[i][j] = B[i][j];
//             buff_C[i][j] = C[i][j];
//             buff_D[i][j] = D[i][j];
//             buff_E_out[i][j] = 0;
//             tmp1[i][j] = 0;
//             tmp2[i][j] = 0;
//         }
//     }
// 
//     lp1: for (i = 0; i < N; i++) {
//         lp2: for (j = 0; j < N; j++) {
//             lp3: for (k = 0; k < N; k++) {
//#pragma HLS unroll factor=1
//                 tmp1[i][j] += buff_A[i][k] * buff_B[k][j];
//             }
//         }
//     }
// 
//     lp4: for (i = 0; i < N; i++) {
//         lp5: for (j = 0; j < N; j++) {
//             lp6: for (k = 0; k < N; k++) {
//#pragma HLS unroll factor=1
//                 tmp2[i][j] += buff_C[i][k] * buff_D[k][j];
//             }
//         }
//     }
//     
//     lp7: for (i = 0; i < N; i++) {
//         lp8: for (j = 0; j < N; j++) {
//             lp9: for (k = 0; k < N; k++) {
//#pragma HLS unroll factor=1
//                 buff_E_out[i][j] += tmp1[i][k] * tmp2[k][j];
//             }
//         }
//     }
// 
//     lpwr_1: for (i = 0; i < N; i++) {
//         lpwr_2: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
//             E_out[i][j] = buff_E_out[i][j];
//         }
//     }
// }

