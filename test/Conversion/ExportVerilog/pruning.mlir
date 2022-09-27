// RUN: circt-opt %s --export-verilog --verify-diagnostics -o %t | FileCheck %s --strict-whitespace

// CHECK-LABEL: module zeroWidthPAssign(
// CHECK:       always_ff @(posedge clk) begin        
// CHECK-NEXT:    // Pruned (Zero Width):     _GEN <= arg0;   
// CHECK-NEXT:  end
hw.module @zeroWidthPAssign(%arg0: i0, %clk: i1) -> (out: i0) {
  %0 = sv.reg  {hw.verilogName = "_GEN"} : !hw.inout<i0>
  sv.alwaysff(posedge %clk) {
    sv.passign %0, %arg0 : i0
  }
  %1 = sv.read_inout %0 : !hw.inout<i0>
  hw.output %1 : i0
}
// CHECK-LABEL: module zeroWidthAssign(
// CHECK:       // Zero width: wire /*Zero Width*/ _GEN;      
// CHECK-NEXT:  // Pruned (Zero Width):   assign _GEN = _GEN; 
// CHECK-NEXT:  // Zero width: assign out = _GEN;     
hw.module @zeroWidthAssign(%arg0: i0, %clk: i1, %a: i0, %b: i1) -> (out: i0) {
  sv.assign %0, %1 : i0
  %0 = sv.wire  {hw.verilogName = "_GEN"} : !hw.inout<i0>
  %1 = sv.read_inout %0 : !hw.inout<i0>
  hw.output %1 : i0
}
