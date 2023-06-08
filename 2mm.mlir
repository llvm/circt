// RUN: circt-opt -convert-affine-to-loopschedule %s 

func.func @kernel_2mm(%alpha: i32, %beta: i32, %tmp: memref<8x8xi32>, %A: memref<8x8xi32>, %B: memref<8x8xi32>, %C: memref<8x8xi32>, %D: memref<8x8xi32>) {
  // %c0_i32 = arith.constant 0 : i32
  // affine.for %i = 0 to 8 {
  //   affine.for %j = 0 to 8 {
  //     // tmp[i][j] = 0
  //     affine.store %c0_i32, %tmp[%i,%j] : memref<8x8xi32>
  //     %sum = affine.for %k = 0 to 8 iter_args(%accum = %c0_i32) -> (i32) {
  //       // tmp[i][j] += alpha * A[i][k] * B[k][j].
  //       %a_value = affine.load %A[%i, %k] : memref<8x8xi32>
  //       %b_value = affine.load %B[%k, %j] : memref<8x8xi32>
        
  //       %0 = arith.muli %a_value, %b_value : i32
  //       %1 = arith.addi %0, %accum : i32
  //       affine.yield %1 : i32
  //     }
  //     %2 = arith.muli %sum, %alpha : i32
  //     affine.store %2, %tmp[%i,%j] : memref<8x8xi32>
  //   }
  // }

  // affine.for %i = 0 to 8 {
  //   affine.for %j = 0 to 8 {
  //     // D[i][j] *= beta 
  //     %d_value = affine.load %D[%i, %j] : memref<8x8xi32>
  //     %0 = arith.muli %d_value, %beta : i32
  //     affine.store %0, %D[%i,%j] : memref<8x8xi32>
  //     affine.for %k = 0 to 8 {
  //       // D[i][j] += tmp[i][k] * C[k][j]
  //       %tmp_value = affine.load %tmp[%i, %k] : memref<8x8xi32>
  //       %c_value = affine.load %C[%k, %j] : memref<8x8xi32>
        
  //       %1 = arith.muli %tmp_value, %c_value : i32
  //       %2 = arith.muli %1, %d_value : i32
  //       affine.store %1, %D[%i,%j] : memref<8x8xi32>
  //     }
  //   }
  // }
  return 
}