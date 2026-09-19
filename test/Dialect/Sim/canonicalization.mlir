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

//===----------------------------------------------------------------------===//
// TriggeredOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @triggered_hoist_single_if(
hw.module @triggered_hoist_single_if(in %clock: !seq.clock, in %enable: i1) {
  %msg = sim.fmt.literal "hello"
  // CHECK: [[MSG:%.+]] = sim.fmt.literal "hello"
  // CHECK: sim.triggered %clock if %enable {
  // CHECK-NEXT: sim.proc.print [[MSG]]
  // CHECK-NEXT: }
  sim.triggered %clock {
    scf.if %enable {
      sim.proc.print %msg
    }
  }
  hw.output
}

// CHECK-LABEL: hw.module @triggered_keep_interleaved_ifs(
hw.module @triggered_keep_interleaved_ifs(in %clock: !seq.clock, in %a: i1, in %b: i1) {
  %a0 = sim.fmt.literal "a0"
  %b0 = sim.fmt.literal "b0"
  %a1 = sim.fmt.literal "a1"
  // CHECK: [[A0:%.+]] = sim.fmt.literal "a0"
  // CHECK: [[B0:%.+]] = sim.fmt.literal "b0"
  // CHECK: [[A1:%.+]] = sim.fmt.literal "a1"
  // CHECK: sim.triggered %clock {
  // CHECK-NEXT: scf.if %a {
  // CHECK-NEXT: sim.proc.print [[A0]]
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.if %b {
  // CHECK-NEXT: sim.proc.print [[B0]]
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.if %a {
  // CHECK-NEXT: sim.proc.print [[A1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  sim.triggered %clock {
    scf.if %a {
      sim.proc.print %a0
    }
    scf.if %b {
      sim.proc.print %b0
    }
    scf.if %a {
      sim.proc.print %a1
    }
  }
  hw.output
}
