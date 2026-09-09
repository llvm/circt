module {
  func.func @_Z9mergesortPiS_(%arg0: memref<16xi32>, %arg1: memref<16xi32>)  {
    %c0 = arith.constant 0 : index
    %c-1_i32 = arith.constant -1 : i32
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = scf.while (%arg2 = %c2_i32) : (i32) -> i32 {
      %1 = arith.cmpi sle, %arg2, %c16_i32 : i32
      scf.condition(%1) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):
      %1 = arith.divsi %c16_i32, %arg2 : i32
      %2 = arith.muli %arg2, %c2_i32 : i32
      %3 = arith.index_cast %1 : i32 to index
      scf.for %arg3 = %c0 to %3 step %c1 {
        %5 = arith.index_cast %arg3 : index to i32
        %6 = arith.muli %arg2, %5 : i32
        %7 = arith.addi %6, %arg2 : i32
        %8 = arith.addi %7, %c-1_i32 : i32
        %9 = arith.muli %2, %5 : i32
        %10 = arith.addi %9, %arg2 : i32
        %11 = arith.addi %10, %c-1_i32 : i32
        %12 = arith.divsi %11, %c2_i32 : i32
        %13 = arith.subi %12, %6 : i32
        %14 = arith.addi %13, %c1_i32 : i32
        %15 = arith.subi %8, %12 : i32
        %16:3 = scf.while (%arg4 = %6, %arg5 = %c0_i32, %arg6 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32) {
          %23 = arith.cmpi slt, %arg6, %14 : i32
          %24 = arith.cmpi slt, %arg5, %15 : i32
          %25 = arith.andi %23, %24 : i1
          scf.condition(%25) %arg5, %arg4, %arg6 : i32, i32, i32
        } do {
        ^bb0(%arg4: i32, %arg5: i32, %arg6: i32):
          %23 = arith.addi %arg6, %6 : i32
          %24 = arith.index_cast %23 : i32 to index
          %25 = memref.load %arg0[%24] : memref<16xi32>
          %26 = arith.addi %arg4, %12 : i32
          %27 = arith.addi %26, %c1_i32 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = memref.load %arg0[%28] : memref<16xi32>
          %30 = arith.cmpi sle, %25, %29 : i32
          %31:2 = scf.if %30 -> (i32, i32) {
            %33 = arith.index_cast %arg5 : i32 to index
            memref.store %25, %arg1[%33] : memref<16xi32>
            %34 = arith.addi %arg6, %c1_i32 : i32
            scf.yield %arg4, %34 : i32, i32
          } else {
            %33 = arith.index_cast %arg5 : i32 to index
            memref.store %29, %arg1[%33] : memref<16xi32>
            %34 = arith.addi %arg4, %c1_i32 : i32
            scf.yield %34, %arg6 : i32, i32
          }
          %32 = arith.addi %arg5, %c1_i32 : i32
          scf.yield %32, %31#0, %31#1 : i32, i32, i32
        }
        %17 = arith.index_cast %14 : i32 to index
        %18 = arith.index_cast %16#2 : i32 to index
        %19:3 = scf.for %arg4 = %18 to %17 step %c1 iter_args(%arg5 = %16#1, %arg6 = %16#2, %arg7 = %16#1) -> (i32, i32, i32) {
          %23 = arith.index_cast %arg5 : i32 to index
          %24 = arith.addi %arg6, %6 : i32
          %25 = arith.index_cast %24 : i32 to index
          %26 = memref.load %arg0[%25] : memref<16xi32>
          memref.store %26, %arg1[%23] : memref<16xi32>
          %27 = arith.addi %arg6, %c1_i32 : i32
          %28 = arith.addi %arg5, %c1_i32 : i32
          scf.yield %28, %27, %28 : i32, i32, i32
        }
        %20 = arith.index_cast %15 : i32 to index
        %21 = arith.index_cast %16#0 : i32 to index
        %22 = arith.index_cast %19#2 : i32 to index
        scf.for %arg4 = %21 to %20 step %c1 {
          %23 = arith.subi %arg4, %21 : index
          %24 = arith.index_cast %arg4 : index to i32
          %25 = arith.addi %22, %23 : index
          %26 = arith.addi %24, %12 : i32
          %27 = arith.addi %26, %c1_i32 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = memref.load %arg0[%28] : memref<16xi32>
          memref.store %29, %arg1[%25] : memref<16xi32>
        }
      }
      affine.for %arg3 = 0 to 16 {
        %5 = affine.load %arg1[%arg3] : memref<16xi32>
        affine.store %5, %arg0[%arg3] : memref<16xi32>
      }
      %4 = arith.muli %arg2, %c2_i32 {polygeist.ssa_names = ["i"]} : i32
      scf.yield %4 : i32
    }
    return
  }
}
