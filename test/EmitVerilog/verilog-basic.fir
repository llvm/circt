; RUN: spt-translate -parse-fir --mlir-print-debuginfo %s | spt-translate -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

circuit Xorr :
  module inputs_only :
    input a: UInt<1>
    input b: UInt<1>

; CHECK-LABEL: module inputs_only(
; CHECK-NEXT:    input  a,
; CHECK-NEXT:    input  b);
; CHECK-NEXT:  endmodule


  module no_ports :
    skip

; CHECK-LABEL: module no_ports();
; CHECK-NEXT:  endmodule


  module Expressions :
    input in4: UInt<4>
    output out1: UInt<1>
    output out4: UInt<4>
    out1 <= xorr(in4)
    out1 <= andr(in4)
    out1 <= orr(in4)
    out1 <= not(in4)
    out4 <= rem(in4, UInt<4>("h1"))

; CHECK-LABEL: module Expressions(
; CHECK-NEXT:     input  [3:0] in4,
; CHECK-NEXT:     output       out1,
; CHECK-NEXT:     output [3:0] out4);
; CHECK-NEXT:     assign out1 = ^in4;
; CHECK-NEXT:     assign out1 = &in4;
; CHECK-NEXT:     assign out1 = |in4;
; CHECK-NEXT:     assign out1 = ~in4;
; CHECK-NEXT:     assign out4 = in4 % 4'h1;
; CHECK-NEXT:  endmodule


  module MultiUseExpr :
    input a: UInt<4>
    output b: UInt<1>
    ;; This disappears in MLIR because of the _T name.  The verilog emitter
    ;; needs to reintroduce it as an 'assign' to a local.
    node _T_42 = andr(xorr(a))
    b <= andr(_T_42)
    b <= xorr(_T_42)

; CHECK-LABEL: module MultiUseExpr(
; CHECK-NEXT:    input  [3:0] a,
; CHECK-NEXT:    output       b);
; CHECK-NEXT:    assign _T = &^a;
; CHECK-NEXT:    assign b = &_T;
; CHECK-NEXT:    assign b = ^_T;
; CHECK-NEXT:  endmodule
