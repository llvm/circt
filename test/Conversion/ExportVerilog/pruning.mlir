// RUN: circt-opt --export-verilog --verify-diagnostics %s -o %t | FileCheck %s --strict-whitespace

// CHECK-LABEL: module zeroWidthPAssign(
// CHECK:       always_ff @(posedge clk) begin        
// CHECK-NEXT:  end
hw.module @zeroWidthPAssign(%arg0: i0, %clk: i1) -> (out: i0) {
  %0 = sv.reg  {hw.verilogName = "_GEN"} : !hw.inout<i0>
  sv.alwaysff(posedge %clk) {
    sv.passign %0, %arg0 : i0
  }
  %1 = sv.read_inout %0 : !hw.inout<i0>
  hw.output %1 : i0
}
// CHECK-LABEL: module zeroWidthLogic(
// CHECK-NOT: reg
hw.module @zeroWidthLogic(%arg0: i0, %sel : i1, %clk: i1) -> (out: i0) {
  %r = sv.reg : !hw.inout<i0>
  %rr = sv.read_inout %r : !hw.inout<i0>
  %2 = comb.mux %sel, %rr, %arg0 : i0
  hw.output %2 : i0
}
