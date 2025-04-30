// RUN: circt-opt --export-verilog --verify-diagnostics %s -o %t | FileCheck %s --strict-whitespace

// CHECK-LABEL: module zeroWidthPAssign(
// CHECK:       always_ff @(posedge clk) begin        
// CHECK-NEXT:  end
hw.module @zeroWidthPAssign(in %arg0 : i0, in %clk: i1, out out: i0) {
  %0 = sv.reg  {hw.verilogName = "_GEN"} : !hw.inout<i0>
  sv.alwaysff(posedge %clk) {
    sv.passign %0, %arg0 : i0
  }
  %1 = sv.read_inout %0 : !hw.inout<i0>
  hw.output %1 : i0
}

// CHECK-LABEL: module zeroWidthLogic(
// CHECK-NOT: reg
hw.module @zeroWidthLogic(in %arg0 : i0, in %sel : i1, in %clk : i1, out out : i0) {
  %r = sv.reg : !hw.inout<i0>
  %rr = sv.read_inout %r : !hw.inout<i0>
  %2 = comb.mux %sel, %rr, %arg0 : i0
  hw.output %2 : i0
}

// CHECK-LABEL: module zeroWidthArith(
// CHECK-NEXT:    // input  /*Zero Width*/ arg0,
// CHECK-NEXT:    //                       arg1,
// CHECK-NEXT:    // output /*Zero Width*/ out
// CHECK-NEXT:  );
hw.module @zeroWidthArith(in %arg0 : i0, in %arg1 : i0, out out : i0) {
  %add = comb.add %arg0, %arg1 : i0
  %sub = comb.sub %arg0, %arg1 : i0
  %and = comb.and %add, %sub : i0
  hw.output %and : i0
}

// CHECK-LABEL: module Concat(
hw.module @Concat(in %arg0 : i0, in %arg1 : i1, in %clk : i1, out out: i2) {
  // CHECK:  assign out = {arg1, clk};
  %1 = comb.concat %arg0, %arg1, %clk : i0, i1, i1
  hw.output %1 : i2
}

// CHECK-LABEL: module icmp(
hw.module @icmp(in %a : i0, out y: i1) {
  // CHECK: assign y = 1'h1;
  %0 = comb.icmp eq %a, %a : i0
  hw.output %0 : i1
}


// CHECK-LABEL: module parity(
hw.module @parity(in %arg0 : i0, out out: i1) {
  // CHECK: assign out = 1'h0;
  %0 = comb.parity %arg0 : i0
  hw.output %0 : i1
}
