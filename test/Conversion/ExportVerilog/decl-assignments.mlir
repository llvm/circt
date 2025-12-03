// RUN: circt-opt --export-verilog %s | FileCheck %s --check-prefixes=CHECK,ALLOW
// RUN: circt-opt --test-apply-lowering-options='options=disallowDeclAssignments' --export-verilog %s | FileCheck %s --check-prefixes=CHECK,DISALLOW

// CHECK-LABEL: module test(
hw.module @test(in %v: i1) {
  // ALLOW:    wire w = v;
  // DISALLOW: wire w;
  // DISALLOW: assign w = v;
  %w = sv.wire : !hw.inout<i1>
  sv.assign %w, %v : i1
  // CHECK: initial begin
  sv.initial {
    // ALLOW:         automatic logic l = v;
    // DISALLOW:      automatic logic l;
    // DISALLOW-NEXT: l = v;
    %l = sv.logic : !hw.inout<i1>
    sv.bpassign %l, %v : i1
  }
}
