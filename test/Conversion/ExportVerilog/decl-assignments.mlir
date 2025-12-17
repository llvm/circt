// RUN: circt-opt --export-verilog %s | FileCheck %s --check-prefixes=CHECK,ALLOW
// RUN: circt-opt --test-apply-lowering-options='options=disallowDeclAssignments' --export-verilog %s | FileCheck %s --check-prefixes=CHECK,DISALLOW

// CHECK-LABEL: module test(
hw.module @test(in %v: i1) {
  // ALLOW:         wire w = v;
  // DISALLOW:      wire w;
  // DISALLOW-NEXT: wire u;
  // DISALLOW-NEXT: wire x;
  // DISALLOW-NEXT: assign w = v;
  // DISALLOW-NEXT: assign u = v;
  %w = sv.wire : !hw.inout<i1>
  sv.assign %w, %v : i1
  %u = sv.wire : !hw.inout<i1>
  sv.assign %u, %v : i1
  // CHECK: initial begin
  sv.initial {
    // ALLOW:         automatic logic l = v;
    // DISALLOW:      automatic logic l;
    // DISALLOW-NEXT: l = v;
    %l = sv.logic : !hw.inout<i1>
    sv.bpassign %l, %v : i1
  }
  // ALLOW:         wire x = v;
  // DISALLOW:      assign x = v;
  %x = sv.wire : !hw.inout<i1>
  sv.assign %x, %v : i1
}
