// RUN: circt-translate %s --export-verilog --verify-diagnostics --lowering-options=alwaysFF | FileCheck %s --strict-whitespace

// CHECK-LABEL: module AlwaysSpill(
hw.module @AlwaysSpill(%port: i1) {
  %false = hw.constant false
  %true = hw.constant true
  %awire = sv.wire : !hw.inout<i1>

  // CHECK: wire [[AWIRE:.+]] = awire;
  %awire2 = sv.read_inout %awire : !hw.inout<i1>

  // Existing simple names should not cause additional spill.
  // CHECK: always @(posedge port)
  sv.always posedge %port {}
  // CHECK: always_ff @(posedge port)
  sv.alwaysff(posedge %port) {}
  // CHECK: always @(posedge [[AWIRE]])
  sv.always posedge %awire2 {}
  // CHECK: always_ff @(posedge [[AWIRE]])
  sv.alwaysff(posedge %awire2) {}

  // Constant values should cause a spill.
  // CHECK: wire [[TMP:.+]] =
  // CHECK-NEXT: always @(posedge [[TMP]])
  sv.always posedge %false {}
  // CHECK: wire [[TMP:.+]] =
  // CHECK-NEXT: always_ff @(posedge [[TMP]])
  sv.alwaysff(posedge %true) {}
}
