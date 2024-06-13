; RUN: firtool %s --format=fir  | FileCheck %s

FIRRTL version 4.0.0
circuit DPI:
; CHECK-LABEL: import "DPI-C" function void clocked_result(
; CHECK-NEXT:   input  [1:0] in_0,
; CHECK-NEXT:                in_1,
; CHECK-NEXT:   output [1:0] out_0
; CHECK-NEXT: );

; CHECK-LABEL: import "DPI-C" function void clocked_void(
; CHECK-NEXT:   input [1:0] in_0,
; CHECK-NEXT:               in_1
; CHECK-NEXT: );

; CHECK-LABEL: import "DPI-C" function void unclocked_result(
; CHECK-NEXT:   input  [1:0] in_0,
; CHECK-NEXT:                in_1,
; CHECK-NEXT:   output [1:0] out_0
; CHECK-NEXT: );

; CHECK-LABEL: module DPI(
; CHECK:        logic [1:0] [[TMP:_.+]];
; CHECK-NEXT:   reg   [1:0] [[RESULT1:_.+]];
; CHECK-NEXT:   always @(posedge clock) begin
; CHECK-NEXT:     if (enable) begin
; CHECK-NEXT:       clocked_result(in_0, in_1, [[TMP]]);
; CHECK-NEXT:       [[RESULT1]] <= [[TMP]];
; CHECK-NEXT:       clocked_void(in_0, in_1);
; CHECK-NEXT:     end
; CHECK-NEXT:   end // always @(posedge)
; CHECK-NEXT:   reg   [1:0] [[RESULT2:_.+]];
; CHECK-NEXT:   always_comb begin
; CHECK-NEXT:     if (enable) begin
; CHECK-NEXT:       unclocked_result(in_0, in_1, [[RESULT2]]);
; CHECK-NEXT:     end
; CHECK-NEXT:     else
; CHECK-NEXT:       [[RESULT2]] = 2'bx;
; CHECK-NEXT:   end // always_comb
; CHECK-NEXT:   assign out_0 = [[RESULT1]];
; CHECK-NEXT:   assign out_1 = [[RESULT2]];
; CHECK-NEXT: endmodule
  public module DPI :
    input clock: Clock
    input enable: UInt<1>
    input in: UInt<2>[2]
    output out : UInt<2>[2]

    node result1 = intrinsic(circt_dpi_call<isClocked = 1, functionName="clocked_result"> : UInt<2>, clock, enable, in[0], in[1])
    intrinsic(circt_dpi_call<isClocked = 1, functionName="clocked_void">, clock, enable, in[0], in[1])
    node result2 = intrinsic(circt_dpi_call<isClocked = 0, functionName="unclocked_result"> : UInt<2>,  enable, in[0], in[1])

    out[0] <= result1
    out[1] <= result2
