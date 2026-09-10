// RUN: hls-est %s --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis | FileCheck %s -check-prefix=CHECK-BRAM

// CHECK-BRAM: Total BRAM: 8

module {
  func.func @kernel(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) {
    %cst = arith.constant 2.000000e+00 : f32
    %alloca = memref.alloca() {hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "cyclic", variable = "buf"}], polygeist.varname = "buf"} : memref<64x64xf32>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %arg1[%arg2, %arg3] : memref<64x64xf32>
        affine.store %0, %alloca[%arg2, %arg3] : memref<64x64xf32>
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %alloca[%arg2, %arg3] : memref<64x64xf32>
        %1 = arith.mulf %0, %cst : f32
        affine.store %1, %alloca[%arg2, %arg3] : memref<64x64xf32>
      }
    }
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %alloca[%arg2, %arg3] : memref<64x64xf32>
        affine.store %0, %arg0[%arg2, %arg3] : memref<64x64xf32>
      }
    }
    return
  }
}
// Kernel:
// #define N 64
// #define M 64
//  
// void kernel(float out[N][M], float in1[N][M]) {
//     float buf[N][M];
//  #pragma HLS array_partition variable=buf cyclic factor=2 dim=1
// 
//     // phase 1: load input into the buffer
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < M; j++) {
//             buf[i][j] = in1[i][j];
//         }
//     }
//  
//     // phase 2: cross-row self-dependence, constant distance 3 in the outer dim,
//     //          zero distance in the inner dim (inner loop is independent -> II=1)
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < M; j++) {
//             buf[i][j] = 2 * buf[i][j];
//         }
//     }
//  
//     // phase 3: drain the buffer to output
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < M; j++) {
//             out[i][j] = buf[i][j];
//         }
//     }
// }