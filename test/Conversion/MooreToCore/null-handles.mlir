// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

moore.class.classdecl @NullClass {
}

// CHECK-LABEL: func.func @NullToChandle() -> !llvm.ptr {
// CHECK-NEXT:    [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    return [[NULL]] : !llvm.ptr
// CHECK-NEXT:  }
func.func @NullToChandle() -> !moore.chandle {
  %null = moore.null
  %0 = moore.conversion %null : !moore.null -> !moore.chandle
  return %0 : !moore.chandle
}

// CHECK-LABEL: func.func @NullToClass() -> !llvm.ptr {
// CHECK-NEXT:    [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    return [[NULL]] : !llvm.ptr
// CHECK-NEXT:  }
func.func @NullToClass() -> !moore.class<@NullClass> {
  %null = moore.null
  %0 = moore.conversion %null : !moore.null -> !moore.class<@NullClass>
  return %0 : !moore.class<@NullClass>
}

// Source event variables lower to one-bit Moore values; assigning null to such
// an event must produce the event's inactive value.
// CHECK-LABEL: func.func @NullToEventBit() -> i1 {
// CHECK:         [[FALSE:%.+]] = hw.constant false
// CHECK:         return [[FALSE]] : i1
// CHECK-NEXT:  }
func.func @NullToEventBit() -> !moore.i1 {
  %null = moore.null
  %0 = moore.conversion %null : !moore.null -> !moore.i1
  return %0 : !moore.i1
}
