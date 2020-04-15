; RUN: spt-translate -parse-fir --mlir-print-debuginfo %s | spt-translate -emit-verilog | FileCheck %s

circuit Xorr :
  module Xorr :
    input a: UInt<4>
    output b: UInt<1>
    b <= xorr(a)

; CHECK: Not implemented yet


; TODO: CECK: module Xorr(
; TODO: CECK:   input  [3:0] a,
; TODO: CECK:   output  b
; TODO: CECK: );
; TODO: CECK:   assign b = ^a;
; TODO: CECK: endmodule


  module Andr :
    input a: UInt<4>
    output b: UInt<1>
    b <= andr(a)

; CHECK: Not implemented yet

; TODO: CECK: module Andr(
; TODO: CECK:   input  [3:0] a,
; TODO: CECK:   output  b
; TODO: CECK: );
; TODO: CECK:   assign b = &a;
; TODO: CECK: endmodule
