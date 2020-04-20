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
    input a: UInt<4>
    output b: UInt<1>
    b <= xorr(a)
    b <= andr(a)
    b <= orr(a)
    b <= not(a)

; CHECK-LABEL: module Expressions(
; CHECK-NEXT:     input  [3:0] a,
; CHECK-NEXT:     output       b);
; CHECK-NEXT:     assign b = ^a;
; CHECK-NEXT:     assign b = &a;
; CHECK-NEXT:     assign b = |a;
; CHECK-NEXT:     assign b = ~a;
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
