; RUN: spt-translate -parse-fir --mlir-print-debuginfo %s | spt-translate -emit-verilog | FileCheck %s

circuit Xorr :
  module Xorr :
    input a: UInt<4>
    output b: UInt<1>
    b <= xorr(a)

; CHECK: module Xorr(
; TODO:    input  [3:0] a,
; TODO:    output  b
; CHECK: );
; TODO:    assign b = ^a;
; CHECK: endmodule


  module Andr :
    input a: UInt<4>
    output b: UInt<1>
    b <= andr(a)

; CHECK: module Andr(
; TODO:    input  [3:0] a,
; TODO:    output  b
; CHECK: );
; TODO:    assign b = &a;
; CHECK: endmodule
