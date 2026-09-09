module {
 func.func @spmv(%arg0: memref<1666xf64>, %arg1: memref<1666xi32>, %arg2: memref<495xi32>, %arg3: memref<494xf64>, %arg4: memref<494xf64>)  {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg5 = 0 to 494 {
      %0 = affine.load %arg2[%arg5] : memref<495xi32>
      %1 = affine.load %arg2[%arg5 + 1] : memref<495xi32>
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %0 : i32 to index
      %4 = scf.for %arg6 = %3 to %2 step %c1 iter_args(%arg7 = %cst) -> (f64) {
        %5 = memref.load %arg0[%arg6] : memref<1666xf64>
        %6 = memref.load %arg1[%arg6] : memref<1666xi32>
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg3[%7] : memref<494xf64>
        %9 = arith.mulf %5, %8 {polygeist.ssa_names = ["Si"]} : f64
        %10 = arith.addf %arg7, %9 {polygeist.ssa_names = ["sum"]} : f64
        scf.yield %10 : f64
      }
      affine.store %4, %arg4[%arg5] : memref<494xf64>
    }
    return
  }
}
