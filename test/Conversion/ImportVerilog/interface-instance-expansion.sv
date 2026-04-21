// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

interface input_if(input logic clk);
  logic seen;
  assign seen = clk;
endinterface

// CHECK-LABEL: moore.module @InterfaceContinuousAssign(
// CHECK-SAME: in %[[CLKA:[^ ,]+]] : !moore.l1, in %[[CLKB:[^ ,]+]] : !moore.l1, out yA : !moore.l1, out yB : !moore.l1
module InterfaceContinuousAssign(input logic clkA, input logic clkB,
                                 output logic yA, output logic yB);
  input_if a(clkA);
  input_if b(clkB);
  assign yA = a.seen;
  assign yB = b.seen;

  // CHECK: %[[A_SEEN:.+]] = moore.assigned_variable %[[CLKA]] : l1
  // CHECK: %[[B_SEEN:.+]] = moore.assigned_variable %[[CLKB]] : l1
  // CHECK: moore.output %[[A_SEEN]], %[[B_SEEN]] : !moore.l1, !moore.l1
endmodule

interface sample_if(input logic clk);
  logic sampled;
  always_comb sampled = clk;
endinterface

// CHECK-LABEL: moore.module @InterfaceProceduralAssign(
// CHECK-SAME: in %[[CLK:[^ ,]+]] : !moore.l1, out y : !moore.l1
module InterfaceProceduralAssign(input logic clk, output logic y);
  sample_if vif(clk);
  assign y = vif.sampled;

  // CHECK: %[[INNER_CLK:.+]] = moore.variable name "clk" : <l1>
  // CHECK: %[[SAMPLED:.+]] = moore.variable : <l1>
  // CHECK: moore.procedure always_comb {
  // CHECK:   %[[CLK_READ:.+]] = moore.read %[[INNER_CLK]] : <l1>
  // CHECK:   moore.blocking_assign %[[SAMPLED]], %[[CLK_READ]] : l1
  // CHECK:   moore.return
  // CHECK: }
  // CHECK: %[[SAMPLED_READ:.+]] = moore.read %[[SAMPLED]] : <l1>
  // CHECK: moore.assign %[[INNER_CLK]], %[[CLK]] : l1
  // CHECK: moore.output %[[SAMPLED_READ]] : !moore.l1
endmodule
