module {
    func.func @stencil(%arg0: memref<8192xi32>, %arg1: memref<8192xi32>, %arg2: memref<9xi32>)  {
    %c0_i32 = arith.constant {polygeist.ssa_names = ["r"]} 0 : i32
    affine.for %arg3 = 0 to 126 {
      affine.for %arg4 = 0 to 62 {
        %0 = affine.for %arg5 = 0 to 3 iter_args(%arg6 = %c0_i32) -> (i32) {
          %1 = affine.for %arg7 = 0 to 3 iter_args(%arg8 = %arg6) -> (i32) {
            %2 = affine.load %arg2[%arg7 + %arg5 * 3] : memref<9xi32>
            %3 = affine.load %arg0[%arg5 * 64 + %arg7 + %arg4 + %arg3 * 64] : memref<8192xi32>
            %4 = arith.muli %2, %3 {polygeist.ssa_names = ["mul"]} : i32
            %5 = arith.addi %arg8, %4 : i32
            affine.yield %5 : i32
          }
          affine.yield %1 : i32
        }
        affine.store %0, %arg1[%arg4 + %arg3 * 64] : memref<8192xi32>
      }
    }
    return
  }
}
