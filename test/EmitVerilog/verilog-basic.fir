; RUN: spt-translate -parse-fir --mlir-print-debuginfo %s | spt-translate -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

circuit Xorr :
  module inputs_only :
    input a: UInt<1>
    input b: UInt<1>

; CHECK:      module inputs_only(
; CHECK-NEXT:   input  a,
; CHECK-NEXT:  input  b);
; CHECK-NEXT: endmodule

  module no_ports :
    skip   ; expected-error {{cannot emit this operation}}
   ; expected-note @-1 {{}}

; CHECK:      module no_ports();
; CHECK-NEXT: endmodule

  module Xorr :
    input a: UInt<4>
    output b: UInt<1>
    b <= xorr(a)

; CHECK:       module Xorr(
; CHECK-NEXT:    input  [3:0] a,
; CHECK-NEXT:    output       b);
; TODO:          assign b = ^a;
; CHECK-NEXT:    assign x = y;
; CHECK-NEXT: endmodule


  module Andr :
    input a: UInt<4>
    output b: UInt<1>
    b <= andr(a)

; CHECK:      module Andr(
; CHECK-NEXT:   input  [3:0] a,
; CHECK-NEXT:   output       b);


; TODO:         assign b = &a;
; CHECK-NEXT:   assign x = y;
; CHECK-NEXT: endmodule
