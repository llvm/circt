// RUN: circt-verilog --ir-hw %s | FileCheck %s

module InoutWriteIR
    (inout wire c,
     input logic clk,
     output logic q_comb,
     output logic q_ff,
     output logic c_drv);
  logic c_next;

  assign c = c_drv;
  always_comb
    c_next = ~c;
  always_comb
    q_comb = c;

  always_ff @(posedge clk) begin
    q_ff <= c;
    c_drv <= c_next;
  end

  // CHECK-LABEL: hw.module @InoutWriteIR
  // CHECK: %[[CVAL:.*]] = llhd.prb %c : i1
  // CHECK: llhd.drv %c, %[[CVAL]] after %
endmodule
