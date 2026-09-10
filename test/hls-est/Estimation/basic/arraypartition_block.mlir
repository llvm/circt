// RUN: hls-est %s --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis | FileCheck %s -check-prefix=CHECK-BRAM

// CHECK-BRAM: Total BRAM: 4

module {
  func.func @kernel(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() {hls.array_partition = [{dim = 1 : i32, kind = "block", factor = 2 : i32, variable = "buf"}], polygeist.varname = "buf"} : memref<1024xi32>
    affine.for %arg2 = 0 to 1024 {
      %0 = affine.load %arg0[%arg2] : memref<1024xi32>
      affine.store %0, %alloca[%arg2] : memref<1024xi32>
    }
    affine.for %arg2 = 0 to 1022 step 2 {
      %0 = affine.load %alloca[%arg2] : memref<1024xi32>
      %1 = arith.addi %0, %c1_i32 : i32
      affine.store %1, %alloca[%arg2] : memref<1024xi32>
      %2 = affine.load %alloca[%arg2 + 1] : memref<1024xi32>
      %3 = arith.muli %2, %c2_i32 : i32
      affine.store %3, %alloca[%arg2 + 1] : memref<1024xi32>
    }
    affine.for %arg2 = 0 to 1024 {
      %0 = affine.load %alloca[%arg2] : memref<1024xi32>
      affine.store %0, %arg1[%arg2] : memref<1024xi32>
    }
    return
  }
}

// Kernel source:
// #define N 1024
// void kernel(const int in[N], int out[N]) {
//   int buf[N];
// #pragma HLS array_partition variable=buf block factor=2 dim=1
//   for (int i = 0; i < N; i++) {
//     buf[i]=in[i];
//   }
//   for (int i = 0; i < N - 2; i += 2) {
//     out[i] = 1 + buf[i]; 
//     out[i + 1] = 2 * buf[i + 1];
//   }
// }



