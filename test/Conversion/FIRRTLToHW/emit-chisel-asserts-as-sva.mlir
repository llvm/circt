// RUN: circt-opt --pass-pipeline='builtin.module(lower-firrtl-to-hw{verification-flavor=sva})' %s | FileCheck %s --check-prefixes=CHECK,SVA
// RUN: circt-opt --pass-pipeline='builtin.module(lower-firrtl-to-hw{verification-flavor=if-else-fatal})' %s | FileCheck %s --check-prefixes=CHECK,IF_ELSE_FATAL
// RUN: circt-opt --pass-pipeline='builtin.module(lower-firrtl-to-hw{verification-flavor=immediate})' %s | FileCheck %s --check-prefixes=CHECK,IMMEDIATE


firrtl.circuit "ifElseFatalToSVA" {
  // CHECK-LABEL: hw.module @ifElseFatalToSVA
  firrtl.module @ifElseFatalToSVA(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>
  ) {
    firrtl.assert %clock, %cond, %enable, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, format = "ifElseFatal"}
    firrtl.assume %clock, %cond, %enable, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, guards = ["USE_PROPERTY_AS_CONSTRAINT"]}
    // CHECK-NEXT: [[CLK:%.+]] = seq.from_clock %clock
    // SVA: sv.assert.concurrent posedge [[CLK]], {{%.+}} message "assert0"

    // IF_ELSE_FATAL:       sv.always posedge [[CLK]] {
    // IF_ELSE_FATAL-NEXT:    sv.if {{%.+}} {
    // IF_ELSE_FATAL-NEXT:      %ASSERT_VERBOSE_COND_ = sv.macro.ref.expr @ASSERT_VERBOSE_COND_()
    // IF_ELSE_FATAL-NEXT:      sv.if %ASSERT_VERBOSE_COND_ {
    // IF_ELSE_FATAL-NEXT:        sv.error.procedural "assert0"
    // IF_ELSE_FATAL-NEXT:      }

    // IMMEDIATE:      sv.always posedge [[CLK]] {
    // IMMEDIATE-NEXT:   sv.if %enable {
    // IMMEDIATE-NEXT:    sv.assert %cond, immediate message "assert0"

    // CHECK: sv.ifdef @USE_PROPERTY_AS_CONSTRAINT {
    // SVA:           sv.assume.concurrent posedge [[CLK]], {{%.+}}
    // IF_ELSE_FATAL: sv.assume.concurrent posedge [[CLK]], {{%.+}}
    // IMMEDIATE:     sv.assume {{%.+}}
  }

  // Test that an immediate assertion is converted to an assertion with a specified flavor.
  //
  // CHECK-LABEL: hw.module @immediateToConcurrent
  firrtl.module @immediateToConcurrent(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>
  ) {
    firrtl.assert %clock, %cond, %enable, "assert1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    // SVA: sv.assert.concurrent
    // IF_ELSE_FATAL: sv.if
    // IMMEDIATE: sv.assert
  }
}
