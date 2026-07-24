// An active-low asynchronous seq.firreg nested under an sv.ifdef must lower to
// valid IR. The module-level initial block forces the reset value under
// `if (!reset)`; the inverted condition must be materialized in the module body
// (not at the register, which sits inside the ifdef), so it dominates its use.
// Regression for a use-outside-defining-region. Re-running circt-opt verifies
// the lowered IR.
// RUN: circt-opt %s --lower-seq-to-sv | circt-opt | FileCheck %s

sv.macro.decl @SOME_MACRO

// CHECK-LABEL: hw.module @ActiveLowAsyncUnderIfDef
hw.module @ActiveLowAsyncUnderIfDef(in %clk : !seq.clock, in %rst : i1, in %d : i32) {
  %c0 = hw.constant 0 : i32
  // The inverted reset used by the initial block is created in the module body,
  // before the ifdef that contains the register.
  // CHECK: %[[INV:.+]] = comb.xor %rst, %{{.+}} : i1
  // CHECK: sv.ifdef @SOME_MACRO
  sv.ifdef @SOME_MACRO {
    %r = seq.firreg %d clock %clk reset async %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 1 : i32} : i32
  }
  // The module-level initial block forces the reset value under the module-body
  // inversion.
  // CHECK: sv.initial
  // CHECK: sv.if %[[INV]] {
}
