// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @HasBeenReset
hw.module @HasBeenReset(in %clock: i1, in %reset: i1) {
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %true = hw.constant true
  %false = hw.constant false
  %true = hw.constant true

  // CHECK-NEXT: %constResetA0 = hw.wire %false
  // CHECK-NEXT: %constResetA1 = hw.wire %false
  // CHECK-NEXT: %constResetS0 = hw.wire %false
  // CHECK-NEXT: %constResetS1 = hw.wire %false
  %r0 = verif.has_been_reset %clock, async %false
  %r1 = verif.has_been_reset %clock, async %true
  %r2 = verif.has_been_reset %clock, sync %false
  %r3 = verif.has_been_reset %clock, sync %true
  %constResetA0 = hw.wire %r0 sym @constResetA0 : i1
  %constResetA1 = hw.wire %r1 sym @constResetA1 : i1
  %constResetS0 = hw.wire %r2 sym @constResetS0 : i1
  %constResetS1 = hw.wire %r3 sym @constResetS1 : i1

  // CHECK-NEXT: [[TMP1:%.+]] = verif.has_been_reset %false, async %reset
  // CHECK-NEXT: [[TMP2:%.+]] = verif.has_been_reset %true, async %reset
  // CHECK-NEXT: %constClockA0 = hw.wire [[TMP1]]
  // CHECK-NEXT: %constClockA1 = hw.wire [[TMP2]]
  // CHECK-NEXT: %constClockS0 = hw.wire %false
  // CHECK-NEXT: %constClockS1 = hw.wire %false
  %c0 = verif.has_been_reset %false, async %reset
  %c1 = verif.has_been_reset %true, async %reset
  %c2 = verif.has_been_reset %false, sync %reset
  %c3 = verif.has_been_reset %true, sync %reset
  %constClockA0 = hw.wire %c0 sym @constClockA0 : i1
  %constClockA1 = hw.wire %c1 sym @constClockA1 : i1
  %constClockS0 = hw.wire %c2 sym @constClockS0 : i1
  %constClockS1 = hw.wire %c3 sym @constClockS1 : i1
}

// CHECK-LABEL: @clockedAssert
hw.module @clockedAssert(in %clock : i1, in %a : i1, in %en : i1) {
  // CHECK: verif.clocked_assert %a if %en, posedge %clock : i1
  %clk = ltl.clock %a, posedge %clock : i1
  verif.assert %clk if %en : !ltl.sequence
}

// CHECK-LABEL: @clockedAssume
hw.module @clockedAssume(in %clock : i1, in %a : i1, in %en : i1) {
  // CHECK: verif.clocked_assume %a if %en, posedge %clock : i1
  %clk = ltl.clock %a, posedge %clock : i1
  verif.assume %clk if %en : !ltl.sequence
}

// CHECK-LABEL: @clockedCover
hw.module @clockedCover(in %clock : i1, in %a : i1, in %en : i1) {
  // CHECK: verif.clocked_cover %a if %en,  posedge %clock : i1
  %clk = ltl.clock %a, posedge %clock : i1
  verif.cover %clk if %en : !ltl.sequence
}

// CHECK-LABEL: @RemoveUnusedSymbolicValues
hw.module @RemoveUnusedSymbolicValues() {
  // CHECK-NOT: verif.symbolic_value
  // CHECK: hw.output
  %0 = verif.symbolic_value : i32
}

// CHECK-LABEL: @AssertEnableTrue
hw.module @AssertEnableTrue(in %a : i1) {
  %true = hw.constant true
  // CHECK: verif.assert
  // CHECK-NOT: if
  verif.assert %a if %true : i1
  // CHECK: hw.output
}

// CHECK-LABEL: @AssertEnableFalse
hw.module @AssertEnableFalse(in %a : i1) {
  %false = hw.constant false
  // CHECK-NOT: verif.assert
  // CHECK: hw.output
  verif.assert %a if %false : i1
}

// CHECK-LABEL: @AssertBooleanConstantTrue
hw.module @AssertBooleanConstantTrue() {
  %prop = ltl.boolean_constant true
  // CHECK-NOT: verif.assert
  verif.assert %prop : !ltl.property
  // CHECK: hw.output
}

// CHECK-LABEL: @AssumeEnableTrue
hw.module @AssumeEnableTrue(in %a : i1) {
  %true = hw.constant true
  // CHECK: verif.assume
  // CHECK-NOT: if
  verif.assume %a if %true : i1
  // CHECK: hw.output
}

// CHECK-LABEL: @AssumeEnableFalse
hw.module @AssumeEnableFalse(in %a : i1) {
  %false = hw.constant false
  // CHECK-NOT: verif.assume
  // CHECK: hw.output
  verif.assume %a if %false : i1
}

// CHECK-LABEL: @AssumeBooleanConstantTrue
hw.module @AssumeBooleanConstantTrue() {
  %prop = ltl.boolean_constant true
  // CHECK-NOT: verif.assume
  verif.assume %prop : !ltl.property
  // CHECK: hw.output
}

// CHECK-LABEL: @CoverEnableTrue
hw.module @CoverEnableTrue(in %a : i1) {
  %true = hw.constant true
  // CHECK: verif.cover
  // CHECK-NOT: if
  verif.cover %a if %true : i1
  // CHECK: hw.output
}

// Cover operations are NOT canonicalized like asserts and assumes.  Covers are
// part of the verification contract around a module and serve as documentation
// of what properties are expected to be exercised during verification.  Even
// if a cover is trivially true or has a constant enable condition, it should
// be preserved because:
// 1. It documents the verification intent.
// 2. It may be used by verification tools to track coverage metrics.
// 3. Removing "trivial" covers would silently change the verification
//    contract.

// CHECK-LABEL: @CoverEnableFalse
hw.module @CoverEnableFalse(in %a : i1) {
  %false = hw.constant false
  // CHECK: verif.cover
  // CHECK-SAME: if %false
  verif.cover %a if %false : i1
  // CHECK: hw.output
}

// CHECK-LABEL: @CoverBooleanConstantTrue
hw.module @CoverBooleanConstantTrue() {
  %prop = ltl.boolean_constant true
  // CHECK: verif.cover
  verif.cover %prop : !ltl.property
  // CHECK: hw.output
}

// CHECK-LABEL: @ClockedAssertEnableTrue
hw.module @ClockedAssertEnableTrue(in %clock : i1, in %a : i1) {
  %true = hw.constant true
  // CHECK: verif.clocked_assert
  // CHECK-NOT: if
  verif.clocked_assert %a if %true, posedge %clock : i1
  // CHECK: hw.output
}

// CHECK-LABEL: @ClockedAssertEnableFalse
hw.module @ClockedAssertEnableFalse(in %clock : i1, in %a : i1) {
  %false = hw.constant false
  // CHECK-NOT: verif.clocked_assert
  // CHECK: hw.output
  verif.clocked_assert %a if %false, posedge %clock : i1
}

// CHECK-LABEL: @ClockedAssertBooleanConstantTrue
hw.module @ClockedAssertBooleanConstantTrue(in %clock : i1) {
  %prop = ltl.boolean_constant true
  // CHECK-NOT: verif.clocked_assert
  verif.clocked_assert %prop, posedge %clock : !ltl.property
  // CHECK: hw.output
}

// CHECK-LABEL: @ClockedAssumeEnableTrue
hw.module @ClockedAssumeEnableTrue(in %clock : i1, in %a : i1) {
  %true = hw.constant true
  // CHECK: verif.clocked_assume
  // CHECK-NOT: if
  verif.clocked_assume %a if %true, posedge %clock : i1
  // CHECK: hw.output
}

// CHECK-LABEL: @ClockedAssumeEnableFalse
hw.module @ClockedAssumeEnableFalse(in %clock : i1, in %a : i1) {
  %false = hw.constant false
  // CHECK-NOT: verif.clocked_assume
  // CHECK: hw.output
  verif.clocked_assume %a if %false, posedge %clock : i1
}

// CHECK-LABEL: @ClockedAssumeBooleanConstantTrue
hw.module @ClockedAssumeBooleanConstantTrue(in %clock : i1) {
  %prop = ltl.boolean_constant true
  // CHECK-NOT: verif.clocked_assume
  verif.clocked_assume %prop, posedge %clock : !ltl.property
  // CHECK: hw.output
}

// Clocked covers are NOT canonicalized like clocked asserts and assumes for the
// same reasons as regular covers.  See the comment above for details.

// CHECK-LABEL: @ClockedCoverEnableTrue
hw.module @ClockedCoverEnableTrue(in %clock : i1, in %a : i1) {
  %true = hw.constant true
  // CHECK: verif.clocked_cover
  // CHECK-NOT: if
  verif.clocked_cover %a if %true, posedge %clock : i1
  // CHECK: hw.output
}

// CHECK-LABEL: @ClockedCoverEnableFalse
hw.module @ClockedCoverEnableFalse(in %clock : i1, in %a : i1) {
  %false = hw.constant false
  // CHECK: verif.clocked_cover
  // CHECK-SAME: if %false
  verif.clocked_cover %a if %false, posedge %clock : i1
  // CHECK: hw.output
}

// CHECK-LABEL: @ClockedCoverBooleanConstantTrue
hw.module @ClockedCoverBooleanConstantTrue(in %clock : i1) {
  %prop = ltl.boolean_constant true
  // CHECK: verif.clocked_cover
  verif.clocked_cover %prop, posedge %clock : !ltl.property
  // CHECK: hw.output
}
