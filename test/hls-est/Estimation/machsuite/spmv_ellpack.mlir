module {
  func.func @ellpack(%arg0: memref<4940xf64>, %arg1: memref<4940xi32>, %arg2: memref<494xf64>, %arg3: memref<494xf64>)  {
    affine.for %arg4 = 0 to 494 {
      %0 = affine.load %arg3[%arg4] : memref<494xf64>
      %1 = affine.for %arg5 = 0 to 10 iter_args(%arg6 = %0) -> (f64) {
        %2 = affine.load %arg0[%arg5 + %arg4 * 10] : memref<4940xf64>
        %3 = affine.load %arg1[%arg5 + %arg4 * 10] : memref<4940xi32>
        %4 = arith.index_cast %3 : i32 to index
        %5 = memref.load %arg2[%4] : memref<494xf64>
        %6 = arith.mulf %2, %5 {polygeist.ssa_names = ["Si"]} : f64
        %7 = arith.addf %arg6, %6 : f64
        affine.yield %7 : f64
      }
      affine.store %1, %arg3[%arg4] : memref<494xf64>
    }
    return
  }
}
