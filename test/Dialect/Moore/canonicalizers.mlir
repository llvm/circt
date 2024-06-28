// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @Casts
func.func @Casts(%arg0: !moore.i1) -> (!moore.i1, !moore.i1) {
  // CHECK-NOT: moore.conversion
  // CHECK-NOT: moore.bool_cast
  %0 = moore.conversion %arg0 : !moore.i1 -> !moore.i1
  %1 = moore.bool_cast %arg0 : !moore.i1 -> !moore.i1
  // CHECK: return %arg0, %arg0
  return %0, %1 : !moore.i1, !moore.i1
}

// CHECK-LABEL: moore.module @SingleAssign
moore.module @SingleAssign() {
  // CHECK-NOT: moore.variable
  // CHECK: %a = moore.assigned_variable %0 : <i32>
  %a = moore.variable : <i32>
  // CHECK: %0 = moore.constant 32 : i32
  %0 = moore.constant 32 : i32
  // CHECK: moore.assign %a, %0 : i32
  moore.assign %a, %0 : i32
  moore.output
}

// CHECK-LABEL: moore.module @MultiAssign
moore.module @MultiAssign() {
  // CHECK-NOT: moore.assigned_variable
  // CHECK: %a = moore.variable : <i32>
  %a = moore.variable : <i32>
  // CHECK: %0 = moore.constant 32 : i32
  %0 = moore.constant 32 : i32
  // CHECK: moore.assign %a, %0 : i32
  moore.assign %a, %0 : i32
  // CHECK: %1 = moore.constant 64 : i32
  %1 = moore.constant 64 : i32
  // CHECK: moore.assign %a, %1 : i32
  moore.assign %a, %1 : i32
  moore.output
}
