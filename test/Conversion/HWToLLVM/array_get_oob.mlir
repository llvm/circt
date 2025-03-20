// RUN: circt-opt %s --convert-hw-to-llvm=array-get-oob-sym="" --split-input-file | FileCheck %s --check-prefix=CHECK-NOHANDLER
// RUN: circt-opt %s --convert-hw-to-llvm=array-get-oob-sym=_fooSym --split-input-file | FileCheck %s --check-prefix=CHECK-WITHHANDLER

// CHECK-NOHANDLER: module
// CHECK-NOHANDLER: llvm.func @pow2_size
// CHECK-NOHANDLER:   [[ALLOC:%.+]] = llvm.alloca
// CHECK-NOHANDLER:   [[ZEXT:%.+]]  = llvm.zext %arg1 : i8 to i9
// CHECK-NOHANDLER:   [[GEP:%.+]]   = llvm.getelementptr [[ALLOC]][0, [[ZEXT]]]
// CHECK-NOHANDLER:   [[LOAD:%.+]]  = llvm.load [[GEP]]
// CHECK-NOHANDLER:   llvm.return [[LOAD]]

// CHECK-WITHHANDLER: module
// CHECK-WITHHANDLER-NOT: fooSym
// CHECK-WITHHANDLER: llvm.func @pow2_size
// CHECK-WITHHANDLER:   [[ALLOC:%.+]] = llvm.alloca
// CHECK-WITHHANDLER:   [[ZEXT:%.+]]  = llvm.zext %arg1 : i8 to i9
// CHECK-WITHHANDLER:   [[GEP:%.+]]   = llvm.getelementptr [[ALLOC]][0, [[ZEXT]]]
// CHECK-WITHHANDLER:   [[LOAD:%.+]]  = llvm.load [[GEP]]
// CHECK-WITHHANDLER:   llvm.return [[LOAD]]

module {
  llvm.func @pow2_size(%array: !llvm.array<256 x i32>, %idx: i8) -> (i32) {
    %cast = builtin.unrealized_conversion_cast %array : !llvm.array<256 x i32> to  !hw.array<256xi32>
    %get = hw.array_get %cast[%idx] : !hw.array<256xi32>, i8
    llvm.return %get : i32
  }
}

// -----

// CHECK-NOHANDLER: module
// CHECK-NOHANDLER: llvm.func @non_pow2_size
// CHECK-NOHANDLER-DAG:   [[ALLOC:%.+]] = llvm.alloca
// CHECK-NOHANDLER-DAG:   [[ZEXT:%.+]]  = llvm.zext %arg1 : i8 to i9
// CHECK-NOHANDLER-DAG:   [[GEP:%.+]]   = llvm.getelementptr [[ALLOC]][0, [[ZEXT]]]
// CHECK-NOHANDLER-DAG:   [[LIMIT:%.+]] = llvm.mlir.constant(255 : i9)
// CHECK-NOHANDLER:       [[COND:%.+]]  = llvm.icmp "uge" [[ZEXT]], [[LIMIT]]
// CHECK-NOHANDLER:       [[SEL:%.+]]   = llvm.select [[COND]], [[ALLOC]], [[GEP]]
// CHECK-NOHANDLER:       [[LOAD:%.+]]  = llvm.load [[SEL]]
// CHECK-NOHANDLER:       llvm.return [[LOAD]]

// CHECK-WITHHANDLER: module
// CHECK-WITHHANDLER-NEXT: llvm.func @_fooSym(!llvm.ptr, i64, i32, !llvm.ptr, i64) -> !llvm.ptr
// CHECK-WITHHANDLER-NEXT: llvm.func @non_pow2_size
// CHECK-WITHHANDLER-DAG:   [[ZEXT:%.+]]  = llvm.zext %arg1 : i8 to i9
// CHECK-WITHHANDLER-DAG:   [[LIMIT:%.+]] = llvm.mlir.constant(255 : i9)
// CHECK-WITHHANDLER:       [[COND:%.+]]  = llvm.icmp "uge" [[ZEXT]], [[LIMIT]]
// CHECK-WITHHANDLER:       llvm.cond_br [[COND]]
// CHECK-WITHHANDLER-NEXT:  ^bb1:
// CHECK-WITHHANDLER:       [[FIX:%.+]]   = llvm.call @_fooSym
// CHECK-WITHHANDLER:       llvm.br ^bb2([[FIX]]
// CHECK-WITHHANDLER-NEXT:  ^bb2([[PTR:%.+]]: !llvm.ptr)
// CHECK-WITHHANDLER:       [[LOAD:%.+]]  = llvm.load [[PTR]]
// CHECK-WITHHANDLER:       llvm.return [[LOAD]]

module {
  llvm.func @non_pow2_size(%array: !llvm.array<255 x i32>, %idx: i8) -> (i32) {
    %cast = builtin.unrealized_conversion_cast %array : !llvm.array<255 x i32> to  !hw.array<255xi32>
    %get = hw.array_get %cast[%idx] : !hw.array<255xi32>, i8
    llvm.return %get : i32
  }
}
