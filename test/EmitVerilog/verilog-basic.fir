; RUN: spt-translate -parse-fir --mlir-print-debuginfo %s | spt-translate -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

circuit Xorr :
  module inputs_only :
    input a: UInt<1>
    input b: UInt<1>

; CHECK-LABEL: module inputs_only(
; CHECK-NEXT:    input  a,
; CHECK-NEXT:    input  b);
; CHECK-EMPTY:
; CHECK-NEXT:  endmodule


  module no_ports :
    skip

; CHECK-LABEL: module no_ports();
; CHECK-NEXT:  endmodule


  module Expressions :
    input in4: UInt<4>
    input clock: Clock
    output out1: UInt<1>
    output out4: UInt<4>
    out1 <= xorr(in4)
    out1 <= andr(in4)
    out1 <= orr(in4)
    out1 <= not(in4)
    out4 <= rem(in4, UInt<4>("h1"))
    out1 <= asUInt(clock)
    out1 <= asUInt(asClock(UInt<1>("h1")))

; CHECK-LABEL: module Expressions(
; CHECK-NEXT:     input  [3:0] in4,
; CHECK-NEXT:     input        clock,
; CHECK-NEXT:     output       out1,
; CHECK-NEXT:     output [3:0] out4);
; CHECK-EMPTY:
; CHECK-NEXT:     assign out1 = ^in4;
; CHECK-NEXT:     assign out1 = &in4;
; CHECK-NEXT:     assign out1 = |in4;
; CHECK-NEXT:     assign out1 = ~in4;
; CHECK-NEXT:     assign out4 = in4 % 4'h1;
; CHECK-NEXT:     assign out1 = clock;
; CHECK-NEXT:     assign out1 = 1'h1;
; CHECK-NEXT:  endmodule


  module Precedence :
    input a: UInt<4>
    input b: UInt<4>
    input c: UInt<4>
    output out1: UInt<1>
    output out: UInt<4>
    out <= add(a, mul(b, c))
    out <= add(mul(a, b), c)
    out <= mul(add(a, b), c)
    out <= mul(a, add(b, c))
    out <= mul(add(a, b), add(b, c))
    out1 <= xorr(add(b, c))
    out1 <= or(lt(b, c), gt(b, c))
    out1 <= and(xor(b, c), or(out1, out1))

; CHECK-LABEL: module Precedence(
; CHECK-NEXT:     input  [3:0] a,
; CHECK-NEXT:     input  [3:0] b,
; CHECK-NEXT:     input  [3:0] c,
; CHECK-NEXT:     output       out1,
; CHECK-NEXT:     output [3:0] out);
; CHECK-EMPTY:

; CHECK-NEXT:     assign out = a + b * c;
; CHECK-NEXT:     assign out = a * b + c;
; CHECK-NEXT:     assign out = (a + b) * c;
; CHECK-NEXT:     assign out = a * (b + c);
; CHECK-NEXT:     assign out = (a + b) * (b + c);
; CHECK-NEXT:     assign out1 = ^(b + c);
; CHECK-NEXT:     assign out1 = b < c | b > c;
; CHECK-NEXT:     assign out1 = (b ^ c) & (out1 | out1);
; CHECK-NEXT:  endmodule


  module Sign :
    input a: UInt<4>
    input b: UInt<4>
    input c: SInt<4>
    input d: SInt<4>
    output out: UInt<1>
    out <= lt(a, b)
    out <= lt(c, d)

; CHECK-LABEL: module Sign(
; CHECK-NEXT:     input  [3:0] a,
; CHECK-NEXT:     input  [3:0] b,
; CHECK-NEXT:     input  [3:0] c,
; CHECK-NEXT:     input  [3:0] d,
; CHECK-NEXT:     output       out);
; CHECK-EMPTY:

; FIXME: One of the two of these is wrong.
; CHECK-NEXT:     assign out = a < b;
; CHECK-NEXT:     assign out = c < d;
; CHECK-NEXT:  endmodule

  module MultiUseExpr :
    input a: UInt<4>
    output b: UInt<1>
    ;; This disappears in MLIR because of the _T name.  The verilog emitter
    ;; needs to reintroduce it as an 'assign' to a local.
    node _T_42 = andr(xorr(a))
    node _T_99 = add(a, a)
    b <= andr(_T_42)
    b <= xorr(_T_42)
    b <= andr(_T_99)
    b <= xorr(_T_99)

; CHECK-LABEL: module MultiUseExpr(
; CHECK-NEXT:    input  [3:0] a,
; CHECK-NEXT:    output       b);
; CHECK-EMPTY:
; CHECK-NEXT:    wire       _T;
; CHECK-NEXT:    wire [4:0] _T_0;
; CHECK-EMPTY:
; CHECK-NEXT:    assign _T = &^a;
; CHECK-NEXT:    assign _T_0 = a + a;
; CHECK-NEXT:    assign b = &_T;
; CHECK-NEXT:    assign b = ^_T;
; CHECK-NEXT:    assign b = &_T_0;
; CHECK-NEXT:    assign b = ^_T_0;
; CHECK-NEXT:  endmodule
