// RUN: circt-opt %s -export-verilog -verify-diagnostics | FileCheck %s
//
// This file checks issues related to lint warnings that have been observed in
// tools.

// Check that the property of an assertion is not inlined.  This avoids a lint
// warning in VCS.  See: https://github.com/llvm/circt/issues/2486
//
// CHECK-LABEL: module AssertPropertyInlined
// CHECK:         assert property (@(posedge clock) a | b)
hw.module @AssertPropertyInlined(%clock: i1, %a: i1, %b: i1) -> (b: i1) {
  %0 = comb.or %a, %b : i1
  sv.assert.concurrent posedge %clock, %0 label "hello" message "world"(%a) : i1
  hw.output %0 : i1
}
