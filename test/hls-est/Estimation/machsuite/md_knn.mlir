module {
  func.func @md_kernel(%arg0: memref<256xf64>, %arg1: memref<256xf64>, %arg2: memref<256xf64>, %arg3: memref<256xf64>, %arg4: memref<256xf64>, %arg5: memref<256xf64>, %arg6: memref<4096xi32>)  {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 1.500000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %cst_2 = arith.constant 0.000000e+00 : f64
    affine.for %arg7 = 0 to 256 {
      %0 = affine.load %arg3[%arg7] : memref<256xf64>
      %1 = affine.load %arg4[%arg7] : memref<256xf64>
      %2 = affine.load %arg5[%arg7] : memref<256xf64>
      %3:3 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %cst_2, %arg10 = %cst_2, %arg11 = %cst_2) -> (f64, f64, f64) {
        %4 = affine.load %arg6[%arg8 + %arg7 * 16] : memref<4096xi32>
        %5 = arith.index_cast %4 : i32 to index
        %6 = memref.load %arg3[%5] : memref<256xf64>
        %7 = memref.load %arg4[%5] : memref<256xf64>
        %8 = memref.load %arg5[%5] : memref<256xf64>
        %9 = arith.subf %0, %6 {polygeist.ssa_names = ["delx"]} : f64
        %10 = arith.subf %1, %7 {polygeist.ssa_names = ["dely"]} : f64
        %11 = arith.subf %2, %8 {polygeist.ssa_names = ["delz"]} : f64
        %12 = arith.mulf %9, %9 : f64
        %13 = arith.mulf %10, %10 : f64
        %14 = arith.addf %12, %13 : f64
        %15 = arith.mulf %11, %11 : f64
        %16 = arith.addf %14, %15 : f64
        %17 = arith.divf %cst_1, %16 {polygeist.ssa_names = ["r2inv"]} : f64
        %18 = arith.mulf %17, %17 : f64
        %19 = arith.mulf %18, %17 {polygeist.ssa_names = ["r6inv"]} : f64
        %20 = arith.mulf %19, %cst_0 : f64
        %21 = arith.subf %20, %cst : f64
        %22 = arith.mulf %19, %21 {polygeist.ssa_names = ["potential"]} : f64
        %23 = arith.mulf %17, %22 {polygeist.ssa_names = ["force"]} : f64
        %24 = arith.mulf %9, %23 : f64
        %25 = arith.addf %arg11, %24 : f64
        %26 = arith.mulf %10, %23 : f64
        %27 = arith.addf %arg10, %26 : f64
        %28 = arith.mulf %11, %23 : f64
        %29 = arith.addf %arg9, %28 : f64
        affine.yield %29, %27, %25 : f64, f64, f64
      }
      affine.store %3#2, %arg0[%arg7] : memref<256xf64>
      affine.store %3#1, %arg1[%arg7] : memref<256xf64>
      affine.store %3#0, %arg2[%arg7] : memref<256xf64>
    }
    return
  }
}
