module {
  func.func @gemm(%arg0: memref<4096xf64>, %arg1: memref<4096xf64>, %arg2: memref<4096xf64>)  {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        %0 = affine.for %arg5 = 0 to 64 iter_args(%arg6 = %cst) -> (f64) {
          %1 = affine.load %arg0[%arg5 + %arg3 * 64] : memref<4096xf64>
          %2 = affine.load %arg1[%arg4 + %arg5 * 64] : memref<4096xf64>
          %3 = arith.mulf %1, %2 {polygeist.ssa_names = ["mult"]} : f64
          %4 = arith.addf %arg6, %3 : f64
          affine.yield %4 : f64
        }
        affine.store %0, %arg2[%arg4 + %arg3 * 64] : memref<4096xf64>
      }
    }
    return
  }
}
