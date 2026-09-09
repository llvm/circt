module {
  func.func @stencil3d(%arg0: memref<2xi32>, %arg1: memref<16384xi32>, %arg2: memref<16384xi32>)  {
    affine.for %arg3 = 0 to 32 {
      affine.for %arg4 = 0 to 16 {
        %0 = affine.load %arg1[%arg4 + %arg3 * 16] : memref<16384xi32>
        affine.store %0, %arg2[%arg4 + %arg3 * 16] : memref<16384xi32>
        %1 = affine.load %arg1[%arg4 + %arg3 * 16 + 15872] : memref<16384xi32>
        affine.store %1, %arg2[%arg4 + %arg3 * 16 + 15872] : memref<16384xi32>
      }
    }
    affine.for %arg3 = 1 to 31 {
      affine.for %arg4 = 0 to 16 {
        %0 = affine.load %arg1[%arg4 + %arg3 * 512] : memref<16384xi32>
        affine.store %0, %arg2[%arg4 + %arg3 * 512] : memref<16384xi32>
        %1 = affine.load %arg1[%arg4 + %arg3 * 512 + 496] : memref<16384xi32>
        affine.store %1, %arg2[%arg4 + %arg3 * 512 + 496] : memref<16384xi32>
      }
    }
    affine.for %arg3 = 1 to 31 {
      affine.for %arg4 = 1 to 31 {
        %0 = affine.load %arg1[%arg4 * 16 + %arg3 * 512] : memref<16384xi32>
        affine.store %0, %arg2[%arg4 * 16 + %arg3 * 512] : memref<16384xi32>
        %1 = affine.load %arg1[%arg4 * 16 + %arg3 * 512 + 15] : memref<16384xi32>
        affine.store %1, %arg2[%arg4 * 16 + %arg3 * 512 + 15] : memref<16384xi32>
      }
    }
    affine.for %arg3 = 1 to 31 {
      affine.for %arg4 = 1 to 31 {
        affine.for %arg5 = 1 to 15 {
          %0 = affine.load %arg1[%arg5 + %arg3 * 512 + %arg4 * 16] : memref<16384xi32>
          %1 = affine.load %arg1[%arg5 + %arg3 * 512 + %arg4 * 16 + 512] : memref<16384xi32>
          %2 = affine.load %arg1[%arg5 + %arg3 * 512 + %arg4 * 16 - 512] : memref<16384xi32>
          %3 = arith.addi %1, %2 : i32
          %4 = affine.load %arg1[%arg5 + %arg3 * 512 + %arg4 * 16 + 16] : memref<16384xi32>
          %5 = arith.addi %3, %4 : i32
          %6 = affine.load %arg1[%arg5 + %arg3 * 512 + %arg4 * 16 - 16] : memref<16384xi32>
          %7 = arith.addi %5, %6 : i32
          %8 = affine.load %arg1[%arg5 + %arg3 * 512 + %arg4 * 16 + 1] : memref<16384xi32>
          %9 = arith.addi %7, %8 : i32
          %10 = affine.load %arg1[%arg5 + %arg3 * 512 + %arg4 * 16 - 1] : memref<16384xi32>
          %11 = arith.addi %9, %10 {polygeist.ssa_names = ["sum1"]} : i32
          %12 = affine.load %arg0[0] : memref<2xi32>
          %13 = arith.muli %0, %12 {polygeist.ssa_names = ["mul0"]} : i32
          %14 = affine.load %arg0[1] : memref<2xi32>
          %15 = arith.muli %11, %14 {polygeist.ssa_names = ["mul1"]} : i32
          %16 = arith.addi %13, %15 : i32
          affine.store %16, %arg2[%arg5 + %arg3 * 512 + %arg4 * 16] : memref<16384xi32>
        }
      }
    }
    return
  }
}
