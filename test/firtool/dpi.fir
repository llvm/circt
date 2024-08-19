; RUN: firtool %s --format=fir  | FileCheck %s

FIRRTL version 4.0.0
circuit DPI:
; CHECK-LABEL: `ifndef __CIRCT_DPI_IMPORT_CLOCKED_RESULT
; CHECK-NEXT: import "DPI-C" context function void clocked_result(
; CHECK-NEXT:   input  byte foo,
; CHECK-NEXT:               bar,
; CHECK-NEXT:   output byte baz
; CHECK-NEXT: );
; CHECK:      `define __CIRCT_DPI_IMPORT_CLOCKED_RESULT
; CHECK-NEXT: `endif

; CHECK-LABEL: `ifndef __CIRCT_DPI_IMPORT_CLOCKED_VOID
; CHECK-NEXT: import "DPI-C" context function void clocked_void(
; CHECK-NEXT:   input byte in_0,
; CHECK-NEXT:              in_1,
; CHECK-NEXT:              in_2[]
; CHECK-NEXT: );
; CHECK:      `define __CIRCT_DPI_IMPORT_CLOCKED_VOID
; CHECK-NEXT: `endif


; CHECK-LABEL: `ifndef __CIRCT_DPI_IMPORT_UNCLOCKED_RESULT
; CHECK-NEXT: import "DPI-C" context function void unclocked_result(
; CHECK-NEXT:   input  byte in_0,
; CHECK-NEXT:                    in_1,
; CHECK-NEXT:   output byte out_0
; CHECK-NEXT: );
; CHECK:      `define __CIRCT_DPI_IMPORT_UNCLOCKED_RESULT
; CHECK-NEXT: `endif

; CHECK-LABEL: module DPI(
; CHECK:        logic [7:0] [[TMP:_.+]];
; CHECK-NEXT:   reg   [7:0] [[RESULT1:_.+]];
; CHECK-NEXT:   wire [7:0] [[OPEN_ARRAY:_.+]][0:1];
; CHECK-NEXT:   assign [[OPEN_ARRAY]] = '{in_0, in_1};
; CHECK-NEXT:   always @(posedge clock) begin
; CHECK-NEXT:     if (enable) begin
; CHECK-NEXT:       clocked_result(in_0, in_1, [[TMP]]);
; CHECK-NEXT:       [[RESULT1]] <= [[TMP]];
; CHECK-NEXT:       clocked_void(in_0, in_1, [[OPEN_ARRAY]]);
; CHECK-NEXT:     end
; CHECK-NEXT:   end // always @(posedge)
; CHECK-NEXT:   reg   [7:0] [[RESULT2:_.+]];
; CHECK-NEXT:   always_comb begin
; CHECK-NEXT:     if (enable) begin
; CHECK-NEXT:       unclocked_result(in_0, in_1, [[RESULT2]]);
; CHECK-NEXT:     end
; CHECK-NEXT:     else
; CHECK-NEXT:       [[RESULT2]] = 8'bx;
; CHECK-NEXT:   end // always_comb
; CHECK-NEXT:   assign out_0 = [[RESULT1]];
; CHECK-NEXT:   assign out_1 = [[RESULT2]];
; CHECK-NEXT: endmodule
  public module DPI :
    input clock: Clock
    input enable: UInt<1>
    input in: UInt<8>[2]
    output out : UInt<8>[2]

    node result1 = intrinsic(circt_dpi_call<isClocked = 1, functionName="clocked_result", inputNames="foo;bar", outputName="baz"> : UInt<8>, clock, enable, in[0], in[1])
    intrinsic(circt_dpi_call<isClocked = 1, functionName="clocked_void">, clock, enable, in[0], in[1], in)
    node result2 = intrinsic(circt_dpi_call<isClocked = 0, functionName="unclocked_result"> : UInt<8>,  enable, in[0], in[1])

    out[0] <= result1
    out[1] <= result2
