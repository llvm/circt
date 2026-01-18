// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @StringOperations
func.func @StringOperations() {
  %s0 = moore.constant_string "a" : i32
  %arg0 = moore.int_to_string %s0 : !moore.i32
  // expected-error @above {{failed to legalize operation 'moore.int_to_string'}}
  %s1 = moore.constant_string "b" : i32
  %arg1 = moore.int_to_string %s1 : !moore.i32
  // expected-error @above {{failed to legalize operation 'moore.int_to_string'}}
  %s2 = moore.constant_string "c" : i32
  %arg2 = moore.int_to_string %s2 : !moore.i32
  // CHECK: [[EMPTY:%.+]] = sim.string.concat ()
  %empty = moore.string.concat ()
  // CHECK: sim.string.concat (%arg0)
  %single = moore.string.concat (%arg0)
  // CHECK: [[TWO:%.+]] = sim.string.concat (%arg0, %arg1)
  %two = moore.string.concat (%arg0, %arg1)
  // CHECK: sim.string.concat (%arg0, %arg1, %arg2)
  %three = moore.string.concat (%arg0, %arg1, %arg2)
  // CHECK: [[NESTED:%.+]] = sim.string.concat ([[TWO]], %arg2)
  %nested = moore.string.concat (%two, %arg2)
  // CHECK: sim.string.length %arg0
  %len1 = moore.string.len %arg0
  // CHECK: sim.string.length %arg1
  %len2 = moore.string.len %arg1
  // CHECK: sim.string.length [[EMPTY]]
  %len_empty = moore.string.len %empty
  // CHECK: sim.string.length [[TWO]]
  %len_concat = moore.string.len %two
  // CHECK: sim.string.length [[NESTED]]
  %len_nested = moore.string.len %nested
  return
}

// CHECK-LABEL: func.func @CurrentTime
func.func @CurrentTime() -> !moore.time {
  // CHECK-NEXT: [[TMP:%.+]] = llhd.current_time
  %0 = moore.builtin.time
  // CHECK-NEXT: return [[TMP]] : !llhd.time
  return %0 : !moore.time
}

// CHECK-LABEL: func.func @TimeConversion
func.func @TimeConversion(%arg0: !moore.l64, %arg1: !moore.time) -> (!moore.time, !moore.l64) {
  // CHECK-NEXT: [[TMP0:%.+]] = llhd.int_to_time %arg0
  %0 = moore.logic_to_time %arg0
  // CHECK-NEXT: [[TMP1:%.+]] = llhd.time_to_int %arg1
  %1 = moore.time_to_logic %arg1
  // CHECK-NEXT: return [[TMP0]], [[TMP1]]
  return %0, %1 : !moore.time, !moore.l64
}

// CHECK-LABEL: func.func @IntToStringConversion
func.func @IntToStringConversion(%arg0: !moore.i45) {
  // CHECK-NEXT: sim.string.int_to_string %arg0 : i45
  moore.int_to_string %arg0 : i45
  return
}

// CHECK-LABEL: func.func @StringOperations
func.func @StringOperations() {
  %s0 = moore.constant_string "a" : i32
  %arg0 = moore.int_to_string %s0 : !moore.i32
  // expected-error @above {{failed to legalize operation 'moore.int_to_string'}}
  %s1 = moore.constant_string "b" : i32
  %arg1 = moore.int_to_string %s1 : !moore.i32
  // expected-error @above {{failed to legalize operation 'moore.int_to_string'}}
  %s2 = moore.constant_string "c" : i32
  %arg2 = moore.int_to_string %s2 : !moore.i32
  // CHECK: [[EMPTY:%.+]] = sim.string.concat ()
  %empty = moore.string.concat ()
  // CHECK: sim.string.concat (%arg0)
  %single = moore.string.concat (%arg0)
  // CHECK: [[TWO:%.+]] = sim.string.concat (%arg0, %arg1)
  %two = moore.string.concat (%arg0, %arg1)
  // CHECK: sim.string.concat (%arg0, %arg1, %arg2)
  %three = moore.string.concat (%arg0, %arg1, %arg2)
  // CHECK: [[NESTED:%.+]] = sim.string.concat ([[TWO]], %arg2)
  %nested = moore.string.concat (%two, %arg2)
  // CHECK: sim.string.length %arg0
  %len1 = moore.string.len %arg0
  // CHECK: sim.string.length %arg1
  %len2 = moore.string.len %arg1
  // CHECK: sim.string.length [[EMPTY]]
  %len_empty = moore.string.len %empty
  // CHECK: sim.string.length [[TWO]]
  %len_concat = moore.string.len %two
  // CHECK: sim.string.length [[NESTED]]
  %len_nested = moore.string.len %nested
  return
}
