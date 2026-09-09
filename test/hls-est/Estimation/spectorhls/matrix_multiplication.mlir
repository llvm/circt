module  {
  func.func @_Z10matrix_mulP5dtypePiS1_(%arg0: memref<?x!llvm.struct<(i32, i64)>>, %arg1: memref<1048576xi32>, %arg2: memref<1048576xi32>)  {
    %c1 = arith.constant 1 : index
    %alloca = memref.alloca() {polygeist.varname = "temp"} : memref<1x!llvm.struct<(i32, i64)>>
    %cast = memref.cast %alloca : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>
    %0 = "polygeist.memref2pointer"(%alloca) : (memref<1x!llvm.struct<(i32, i64)>>) -> !llvm.ptr
    %1 = affine.for %arg3 = 0 to 1024 iter_args(%arg4 = %arg0) -> (memref<?x!llvm.struct<(i32, i64)>>) {
      %2 = affine.for %arg5 = 0 to 1024 iter_args(%arg6 = %arg4) -> (memref<?x!llvm.struct<(i32, i64)>>) {
        %3 = "polygeist.subindex"(%arg6, %c1) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?x!llvm.struct<(i32, i64)>>
        func.call @_ZN5dtypeC1ERKS_(%cast, %arg6) : (memref<?x!llvm.struct<(i32, i64)>>, memref<?x!llvm.struct<(i32, i64)>>) -> ()
        %4 = llvm.load %0 : !llvm.ptr -> i32
        affine.for %arg7 = 0 to 1024 {
          %5 = affine.load %arg1[%arg7 + %arg5 * 1024] : memref<1048576xi32>
          %6 = arith.muli %4, %5 : i32
          %7 = affine.load %arg2[%arg7 + %arg3 * 1024] : memref<1048576xi32>
          %8 = arith.addi %7, %6 : i32
          affine.store %8, %arg2[%arg7 + %arg3 * 1024] : memref<1048576xi32>
        }
        affine.yield %3 : memref<?x!llvm.struct<(i32, i64)>>
      }
      affine.yield %2 : memref<?x!llvm.struct<(i32, i64)>>
    }
    return
  }
  func.func @_ZN5dtypeC1ERKS_(%arg0: memref<?x!llvm.struct<(i32, i64)>>, %arg1: memref<?x!llvm.struct<(i32, i64)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.struct<(i32, i64)>>) -> !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> i32
    %2 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(i32, i64)>>) -> !llvm.ptr
    llvm.store %1, %2 : i32, !llvm.ptr
    %3 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i64)>
    %4 = llvm.load %3 : !llvm.ptr -> i64
    %5 = llvm.getelementptr %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i64)>
    llvm.store %4, %5 : i64, !llvm.ptr
    return
  }
}
