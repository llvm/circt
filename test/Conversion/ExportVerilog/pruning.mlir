// RUN: circt-opt --export-verilog --verify-diagnostics %s -o %t | FileCheck %s --strict-whitespace

// CHECK-LABEL: module zeroWidthPAssign(
// CHECK:       always_ff @(posedge clk) begin        
// CHECK-NEXT:  end
hw.module @zeroWidthPAssign(input %arg0 : i0, input %clk: i1, output out: i0) {
  %0 = sv.reg  {hw.verilogName = "_GEN"} : !hw.inout<i0>
  sv.alwaysff(posedge %clk) {
    sv.passign %0, %arg0 : i0
  }
  %1 = sv.read_inout %0 : !hw.inout<i0>
  hw.output %1 : i0
}
// CHECK-LABEL: module zeroWidthLogic(
// CHECK-NOT: reg
hw.module @zeroWidthLogic(input %arg0 : i0, input %sel : i1, input %clk : i1, output out : i0) {
  %r = sv.reg : !hw.inout<i0>
  %rr = sv.read_inout %r : !hw.inout<i0>
  %2 = comb.mux %sel, %rr, %arg0 : i0
  hw.output %2 : i0
}

// CHECK-LABEL: module Concat(
hw.module @Concat(input %arg0 : i0, input %arg1 : i1, input %clk : i1, output out: i2) {
  // CHECK:  assign out = {arg1, clk};
  %1 = comb.concat %arg0, %arg1, %clk : i0, i1, i1
  hw.output %1 : i2
}

// CHECK-LABEL: module icmp(
hw.module @icmp(input %a : i0, output y: i1) {
  // CHECK: assign y = 1'h1;
  %0 = comb.icmp eq %a, %a : i0
  hw.output %0 : i1
}


// CHECK-LABEL: module parity(
hw.module @parity(input %arg0 : i0, output out: i1) {
  // CHECK: assign out = 1'h0;
  %0 = comb.parity %arg0 : i0
  hw.output %0 : i1
}
