module {
  func.func @bbgemm(%arg0: memref<4096xf64>, %arg1: memref<4096xf64>, %arg2: memref<4096xf64>)  {
    affine.for %arg3 = 0 to 64 step 8 {
      affine.for %arg4 = 0 to 64 step 8 {
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 8 {
            %0 = affine.load %arg0[%arg4 + %arg6 + %arg5 * 64] : memref<4096xf64>
            affine.for %arg7 = 0 to 8 {
              %1 = affine.load %arg1[%arg4 * 64 + %arg3 + %arg7 + %arg6 * 64] : memref<4096xf64>
              %2 = arith.mulf %0, %1 {polygeist.ssa_names = ["mul"]} : f64
              %3 = affine.load %arg2[%arg3 + %arg7 + %arg5 * 64] : memref<4096xf64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %arg2[%arg3 + %arg7 + %arg5 * 64] : memref<4096xf64>
            }
          }
        }
      }
    }
    return
  }
}
