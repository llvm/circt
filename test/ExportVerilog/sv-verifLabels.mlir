// RUN: circt-opt --lowering-options= --export-verilog %s | FileCheck %s --check-prefix=CHECK-OFF
// RUN: circt-opt --lowering-options=verifLabels --export-verilog %s | FileCheck %s --check-prefix=CHECK-ON

hw.module @foo(%clock: i1, %cond: i1) {
  sv.initial {
    // CHECK-OFF: assert(
    // CHECK-OFF: assume(
    // CHECK-OFF: cover(
    // CHECK-ON:  assert_0: assert(
    // CHECK-ON:  assume_1: assume(
    // CHECK-ON:  cover_2: cover(
    sv.assert %cond : i1
    sv.assume %cond : i1
    sv.cover %cond : i1
  }
  // CHECK-OFF: assert property
  // CHECK-OFF: assume property
  // CHECK-OFF: cover property
  // CHECK-ON:  assert_3: assert property
  // CHECK-ON:  assume_4: assume property
  // CHECK-ON:  cover_5: cover property
  sv.assert.concurrent posedge %clock %cond : i1
  sv.assume.concurrent posedge %clock %cond : i1
  sv.cover.concurrent posedge %clock %cond : i1

  // Explicitly labeled ops should keep their label.
  sv.initial {
    // CHECK-OFF: imm_assert: assert(
    // CHECK-ON:  imm_assert: assert(
    // CHECK-OFF: imm_assume: assume(
    // CHECK-ON:  imm_assume: assume(
    // CHECK-OFF: imm_cover: cover(
    // CHECK-ON:  imm_cover: cover(
    sv.assert "imm_assert" %cond : i1
    sv.assume "imm_assume" %cond : i1
    sv.cover "imm_cover" %cond : i1
  }
  // CHECK-OFF: con_assert: assert property
  // CHECK-ON:  con_assert: assert property
  // CHECK-OFF: con_assume: assume property
  // CHECK-ON:  con_assume: assume property
  // CHECK-OFF: con_cover: cover property
  // CHECK-ON:  con_cover: cover property
  sv.assert.concurrent "con_assert" posedge %clock %cond : i1
  sv.assume.concurrent "con_assume" posedge %clock %cond : i1
  sv.cover.concurrent "con_cover" posedge %clock %cond : i1

  // Explicitly labeled ops that conflict with implicit labels should force the
  // implicit labels to change, even if they appear earlier in the output.
  sv.initial {
    // CHECK-OFF: assert_0: assert(
    // CHECK-ON:  assert_0_6: assert(
    // CHECK-OFF: assume_2: assume(
    // CHECK-ON:  assume_2: assume(
    // CHECK-OFF: cover_4: cover(
    // CHECK-ON:  cover_4: cover(
    sv.assert "assert_0" %cond : i1
    sv.assume "assume_2" %cond : i1
    sv.cover "cover_4" %cond : i1
  }
  // CHECK-OFF: assert_6: assert property
  // CHECK-ON:  assert_6: assert property
  // CHECK-OFF: assume_8: assume property
  // CHECK-ON:  assume_8: assume property
  // CHECK-OFF: cover_10: cover property
  // CHECK-ON:  cover_10: cover property
  sv.assert.concurrent "assert_6" posedge %clock %cond : i1
  sv.assume.concurrent "assume_8" posedge %clock %cond : i1
  sv.cover.concurrent "cover_10" posedge %clock %cond : i1
}
