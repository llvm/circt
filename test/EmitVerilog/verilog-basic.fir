; RUN: spt-translate -parse-fir --mlir-print-debuginfo %s | spt-translate -emit-verilog | FileCheck %s --strict-whitespace

circuit Xorr :
  module Xorr :
    input a: UInt<4>
    output b: UInt<1>
    b <= xorr(a)

; CHECK: module Xorr(
; CHECK:    input  [3:0] a,
; CHECK:    output       b
; CHECK: );
; TODO:    assign b = ^a;
; CHECK: endmodule


  module Andr :
    input a: UInt<4>
    output b: UInt<1>
    b <= andr(a)

; CHECK: module Andr(
; CHECK:   input  [3:0] a,
; CHECK:   output       b
; CHECK: );
; TODO:    assign b = &a;
; CHECK: endmodule
