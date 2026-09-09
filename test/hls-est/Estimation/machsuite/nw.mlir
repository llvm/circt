module {
  func.func @needwun(%arg0: memref<128xi8>, %arg1: memref<128xi8>, %arg2: memref<256xi8>, %arg3: memref<256xi8>, %arg4: memref<16641xi32>, %arg5: memref<16641xi8>)  {
    %true = arith.constant true
    %c95_i8 = arith.constant 95 : i8
    %c45_i8 = arith.constant 45 : i8
    %c60_i32 = arith.constant 60 : i32
    %c92_i32 = arith.constant 92 : i32
    %c128_i32 = arith.constant {polygeist.ssa_names = ["a_idx"]} 128 : i32
    %c92_i8 = arith.constant 92 : i8
    %c94_i8 = arith.constant 94 : i8
    %c60_i8 = arith.constant 60 : i8
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c129_i32 = arith.constant 129 : i32
    %c0_i32 = arith.constant {polygeist.ssa_names = ["a_idx"]} 0 : i32
    affine.for %arg6 = 0 to 129 {
      %3 = arith.index_cast %arg6 : index to i32
      %4 = arith.muli %3, %c-1_i32 : i32
      affine.store %4, %arg4[%arg6] : memref<16641xi32>
    }
    affine.for %arg6 = 0 to 129 {
      %3 = arith.index_cast %arg6 : index to i32
      %4 = arith.muli %3, %c-1_i32 : i32
      affine.store %4, %arg4[%arg6 * 129] : memref<16641xi32>
    }
    affine.for %arg6 = 1 to 129 {
      affine.for %arg7 = 1 to 129 {
        %3 = affine.load %arg0[%arg7 - 1] : memref<128xi8>
        %4 = affine.load %arg1[%arg6 - 1] : memref<128xi8>
        %5 = arith.cmpi eq, %3, %4 : i8
        %6 = arith.select %5, %c1_i32, %c-1_i32 : i32
        %7 = affine.load %arg4[%arg7 + %arg6 * 129 - 130] : memref<16641xi32>
        %8 = arith.addi %7, %6 {polygeist.ssa_names = ["up_left"]} : i32
        %9 = affine.load %arg4[%arg7 + %arg6 * 129 - 129] : memref<16641xi32>
        %10 = arith.addi %9, %c-1_i32 {polygeist.ssa_names = ["up"]} : i32
        %11 = affine.load %arg4[%arg7 + %arg6 * 129 - 1] : memref<16641xi32>
        %12 = arith.addi %11, %c-1_i32 {polygeist.ssa_names = ["left"]} : i32
        %13 = arith.cmpi sgt, %10, %12 : i32
        %14 = arith.select %13, %10, %12 : i32
        %15 = arith.cmpi sgt, %8, %14 : i32
        %16 = arith.select %15, %8, %14 : i32
        affine.store %16, %arg4[%arg7 + %arg6 * 129] : memref<16641xi32>
        %17 = arith.cmpi eq, %16, %12 : i32
        scf.if %17 {
          affine.store %c60_i8, %arg5[%arg7 + %arg6 * 129] : memref<16641xi8>
        } else {
          %18 = arith.cmpi eq, %16, %10 : i32
          scf.if %18 {
            affine.store %c94_i8, %arg5[%arg7 + %arg6 * 129] : memref<16641xi8>
          } else {
            affine.store %c92_i8, %arg5[%arg7 + %arg6 * 129] : memref<16641xi8>
          }
        }
      }
    }
    %0:5 = scf.while (%arg6 = %c0_i32, %arg7 = %c0_i32, %arg8 = %c128_i32, %arg9 = %c128_i32) : (i32, i32, i32, i32) -> (i32, i32, i32, i32, i32) {
      %3 = arith.cmpi sgt, %arg9, %c0_i32 : i32
      %4 = scf.if %3 -> (i1) {
        scf.yield %true : i1
      } else {
        %6 = arith.cmpi sgt, %arg8, %c0_i32 : i32
        scf.yield %6 : i1
      }
      %5:5 = scf.if %4 -> (i32, i32, i32, i32, i32) {
        %6 = arith.muli %arg8, %c129_i32 {polygeist.ssa_names = ["r"]} : i32
        %7 = arith.addi %6, %arg9 : i32
        %8 = arith.index_cast %7 : i32 to index
        %9 = memref.load %arg5[%8] : memref<16641xi8>
        %10 = arith.extsi %9 : i8 to i32
        %11 = arith.cmpi eq, %10, %c92_i32 : i32
        %12:4 = scf.if %11 -> (i32, i32, i32, i32) {
          %14 = arith.addi %arg7, %c1_i32 : i32
          %15 = arith.index_cast %arg7 : i32 to index
          %16 = arith.addi %arg9, %c-1_i32 : i32
          %17 = arith.index_cast %16 : i32 to index
          %18 = memref.load %arg0[%17] : memref<128xi8>
          memref.store %18, %arg2[%15] : memref<256xi8>
          %19 = arith.addi %arg6, %c1_i32 : i32
          %20 = arith.index_cast %arg6 : i32 to index
          %21 = arith.addi %arg8, %c-1_i32 : i32
          %22 = arith.index_cast %21 : i32 to index
          %23 = memref.load %arg1[%22] : memref<128xi8>
          memref.store %23, %arg3[%20] : memref<256xi8>
          scf.yield %19, %14, %21, %16 : i32, i32, i32, i32
        } else {
          %14 = memref.load %arg5[%8] : memref<16641xi8>
          %15 = arith.extsi %14 : i8 to i32
          %16 = arith.cmpi eq, %15, %c60_i32 : i32
          %17:4 = scf.if %16 -> (i32, i32, i32, i32) {
            %18 = arith.addi %arg7, %c1_i32 : i32
            %19 = arith.index_cast %arg7 : i32 to index
            %20 = arith.addi %arg9, %c-1_i32 : i32
            %21 = arith.index_cast %20 : i32 to index
            %22 = memref.load %arg0[%21] : memref<128xi8>
            memref.store %22, %arg2[%19] : memref<256xi8>
            %23 = arith.addi %arg6, %c1_i32 : i32
            %24 = arith.index_cast %arg6 : i32 to index
            memref.store %c45_i8, %arg3[%24] : memref<256xi8>
            scf.yield %23, %18, %arg8, %20 : i32, i32, i32, i32
          } else {
            %18 = arith.addi %arg7, %c1_i32 : i32
            %19 = arith.index_cast %arg7 : i32 to index
            memref.store %c45_i8, %arg2[%19] : memref<256xi8>
            %20 = arith.addi %arg6, %c1_i32 : i32
            %21 = arith.index_cast %arg6 : i32 to index
            %22 = arith.addi %arg8, %c-1_i32 : i32
            %23 = arith.index_cast %22 : i32 to index
            %24 = memref.load %arg1[%23] : memref<128xi8>
            memref.store %24, %arg3[%21] : memref<256xi8>
            scf.yield %20, %18, %22, %arg9 : i32, i32, i32, i32
          }
          scf.yield %17#0, %17#1, %17#2, %17#3 : i32, i32, i32, i32
        }
        %13 = llvm.mlir.undef : i32
        scf.yield %12#0, %12#1, %12#2, %12#3, %13 : i32, i32, i32, i32, i32
      } else {
        scf.yield %arg6, %arg7, %arg8, %arg9, %arg7 : i32, i32, i32, i32, i32
      }
      scf.condition(%4) %5#0, %5#1, %5#2, %5#3, %5#4 : i32, i32, i32, i32, i32
    } do {
    ^bb0(%arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32):
      scf.yield %arg6, %arg7, %arg8, %arg9 : i32, i32, i32, i32
    }
    %1 = arith.index_cast %0#4 : i32 to index
    affine.for %arg6 = %1 to 256 {
      affine.store %c95_i8, %arg2[%arg6] : memref<256xi8>
    }
    %2 = arith.index_cast %0#0 : i32 to index
    affine.for %arg6 = %2 to 256 {
      affine.store %c95_i8, %arg3[%arg6] : memref<256xi8>
    }
    return
  }
}
