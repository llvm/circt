// RUN: circt-opt --mem2reg --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @Basic(
func.func @Basic() -> !moore.i42 {
  // CHECK: [[TMP:%.*]] = moore.constant 9001 : i42
  // CHECK-NOT: = moore.variable
  // CHECK-NOT: = moore.blocking_assign
  // CHECK-NOT: = moore.read
  %0 = moore.constant 9001 : i42
  %1 = moore.variable : <i42>
  moore.blocking_assign %1, %0 : i42
  %2 = moore.read %1 : <i42>
  // CHECK: return [[TMP]] : !moore.i42
  return %2 : !moore.i42
}

// CHECK-LABEL: func.func @ControlFlow(
func.func @ControlFlow(%arg0: i1, %arg1: !moore.l8) -> !moore.l8 {
  // CHECK-NOT: moore.variable
  // CHECK: [[DEFAULT:%.+]] = moore.constant hXX : l8
  %0 = moore.variable : <l8>
  // CHECK: cf.cond_br %arg0, ^[[BB1:.+]], ^[[BB2:.+]]([[DEFAULT]] : !moore.l8)
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // CHECK-NOT: moore.blocking_assign
  // CHECK: cf.br ^[[BB2]](%arg1 : !moore.l8)
  moore.blocking_assign %0, %arg1 : l8
  cf.br ^bb2
^bb2:
  // CHECK: ^[[BB2]]([[TMP:%.+]]: !moore.l8):
  // CHECK-NOT: moore.read
  // CHECK: return [[TMP]]
  %1 = moore.read %0 : <l8>
  return %1 : !moore.l8
}

// CHECK-LABEL: func.func @InitialValueDoesNotDominateDefault(
func.func @InitialValueDoesNotDominateDefault() {
  cf.br ^bb1
^bb1:
  %0 = moore.constant 0 : i32
  %1 = moore.variable %0 : <i32>
  cf.br ^bb2
^bb2:
  %2 = moore.read %1 : <i32>
  moore.blocking_assign %1, %2 : i32
  cf.br ^bb1
}
