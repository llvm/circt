// RUN: circt-opt %s --canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// String Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @string_get_constant_fold_valid
func.func @string_get_constant_fold_valid() -> i8 {
  %str = sim.string.literal "Hello"
  %idx = hw.constant 1 : i32
  // CHECK: [[TMP:%.+]] = hw.constant 101 : i8
  %char = sim.string.get %str[%idx]
  // CHECK: return [[TMP]]
  return %char : i8
}

// CHECK-LABEL: func.func @string_get_constant_fold_first
func.func @string_get_constant_fold_first() -> i8 {
  %str = sim.string.literal "Hello"
  %idx = hw.constant 0 : i32
  // CHECK: [[TMP:%.+]] = hw.constant 72 : i8
  %char = sim.string.get %str[%idx]
  // CHECK: return [[TMP]]
  return %char : i8
}

// CHECK-LABEL: func.func @string_get_constant_fold_last
func.func @string_get_constant_fold_last() -> i8 {
  %str = sim.string.literal "Hello"
  %idx = hw.constant 4 : i32
  // CHECK: [[TMP:%.+]] = hw.constant 111 : i8
  %char = sim.string.get %str[%idx]
  // CHECK: return [[TMP]]
  return %char : i8
}

// CHECK-LABEL: func.func @string_get_constant_fold_out_of_bounds_positive
func.func @string_get_constant_fold_out_of_bounds_positive() -> i8 {
  %str = sim.string.literal "Hello"
  %idx = hw.constant 100 : i32
  // CHECK: [[TMP:%.+]] = hw.constant 0 : i8
  %char = sim.string.get %str[%idx]
  // CHECK: return [[TMP]]
  return %char : i8
}

// CHECK-LABEL: func.func @string_get_constant_fold_out_of_bounds_negative
func.func @string_get_constant_fold_out_of_bounds_negative() -> i8 {
  %str = sim.string.literal "Hello"
  %idx = hw.constant -1 : i32
  // CHECK: [[TMP:%.+]] = hw.constant 0 : i8
  %char = sim.string.get %str[%idx]
  // CHECK: return [[TMP]]
  return %char : i8
}

// CHECK-LABEL: func.func @string_get_constant_fold_empty_string
func.func @string_get_constant_fold_empty_string() -> i8 {
  %str = sim.string.literal ""
  %idx = hw.constant 0 : i32
  // CHECK: [[TMP:%.+]] = hw.constant 0 : i8
  %char = sim.string.get %str[%idx]
  // CHECK: return [[TMP]]
  return %char : i8
}

// CHECK-LABEL: func.func @string_get_special_chars
func.func @string_get_special_chars() -> i8 {
  %str = sim.string.literal "abc\n"
  %idx = hw.constant 3 : i32
  // CHECK: [[TMP:%.+]] = hw.constant 10 : i8
  %char = sim.string.get %str[%idx]
  // CHECK: return [[TMP]]
  return %char : i8
}
