// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lint)))' --verify-diagnostics --split-input-file %s | FileCheck %s

firrtl.circuit "lint_tests" {
  // CHECK: firrtl.module @lint_tests
  firrtl.module @lint_tests(in %en: !firrtl.uint<1>, in %pred: !firrtl.uint<1>, in %reset: !firrtl.reset, in %clock: !firrtl.clock) {
    %0 = firrtl.asUInt %reset : (!firrtl.reset) -> !firrtl.uint<1>
    // CHECK: firrtl.assert
    firrtl.assert %clock, %pred, %en, "valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>  {eventControl = 0 : i32, isConcurrent = false}
    // CHECK: firrtl.assert
    firrtl.assert %clock, %0, %en, "valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>  {eventControl = 0 : i32, isConcurrent = false}
    %false = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: firrtl.assert
    firrtl.assert %clock, %false, %en, "valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>  {eventControl = 0 : i32, isConcurrent = false}
    // CHECK: firrtl.int.verif.assert
    firrtl.int.verif.assert %pred : !firrtl.uint<1>
    // CHECK: firrtl.int.verif.assert
    firrtl.when %en : !firrtl.uint<1> {
      firrtl.int.verif.assert %false : !firrtl.uint<1>
    }
  }
}

// -----

firrtl.circuit "assert_const" {
  firrtl.module @assert_const(in %clock: !firrtl.clock) {
    %true = firrtl.constant 1 : !firrtl.uint<1>
    // expected-note @below {{constant defined here}}
    %false = firrtl.constant 0 : !firrtl.uint<1>
    // expected-error @below {{'firrtl.assert' op is guaranteed to fail simulation, as the predicate is constant false}}
    firrtl.assert %clock, %false, %true, "valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>  {eventControl = 0 : i32, isConcurrent = false}
    // expected-error @below {{'firrtl.assert' op is guaranteed to fail simulation, as the predicate is constant false}}
    firrtl.assert %clock, %false, %true, "valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>  {eventControl = 0 : i32, isConcurrent = false}
  }
}

// -----

firrtl.circuit "assert_reset" {
  // expected-note @below {{reset signal defined here}}
  firrtl.module @assert_reset(in %en: !firrtl.uint<1>, in %pred: !firrtl.uint<1>, in %reset: !firrtl.reset, in %reset_async: !firrtl.asyncreset, in %clock: !firrtl.clock) {
    %0 = firrtl.asUInt %reset : (!firrtl.reset) -> !firrtl.uint<1>
    %true = firrtl.constant 1 : !firrtl.uint<1>
    %false = firrtl.constant 0 : !firrtl.uint<1>
    // expected-error @below {{op is guaranteed to fail simulation, as the predicate is a reset signal}}
    firrtl.assert %clock, %0, %true, "valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>  {eventControl = 0 : i32, isConcurrent = false}
  }
}

// -----

firrtl.circuit "assert_const2" {
  firrtl.module @assert_const2() {
    // expected-note @below {{constant defined here}}
    %false = firrtl.constant 0 : !firrtl.uint<1>
    // expected-error @below {{op is guaranteed to fail simulation, as the predicate is constant false}}
    firrtl.int.verif.assert %false : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "assert_reset2" {
  // expected-note @below {{reset signal defined here}}
  firrtl.module @assert_reset2(in %en: !firrtl.uint<1>, in %pred: !firrtl.uint<1>, in %reset: !firrtl.reset, in %reset_async: !firrtl.asyncreset, in %clock: !firrtl.clock) {
    %0 = firrtl.asUInt %reset : (!firrtl.reset) -> !firrtl.uint<1>
    // expected-error @below {{op is guaranteed to fail simulation, as the predicate is a reset signal}}
    firrtl.int.verif.assert %0 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "assert_reset3" {
firrtl.layer @GroupFoo bind {}
  // expected-note @below {{reset signal defined here}}
  firrtl.module @assert_reset3(in %en: !firrtl.uint<1>, in %pred: !firrtl.uint<1>, in %reset: !firrtl.reset, in %reset_async: !firrtl.asyncreset, in %clock: !firrtl.clock) {
    %0 = firrtl.asUInt %reset : (!firrtl.reset) -> !firrtl.uint<1>
    firrtl.layerblock @GroupFoo {
      // expected-error @below {{op is guaranteed to fail simulation, as the predicate is a reset signal}}
      firrtl.int.verif.assert %0 : !firrtl.uint<1>
    }
  }
}
