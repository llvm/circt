; RUN: firtool %s --format=fir -verilog | FileCheck %s

; Test that reduction operators with multiple uses are not duplicated.

FIRRTL version 7.0.0

circuit Foo :
  public module Foo :
    input x : UInt<8>
    input y : UInt<8>
    output out : UInt<6>

    node parity = xorr(x)
    node reduce_and = andr(y)
    node reduce_or = orr(y)

    ; Use parity twice
    connect out, cat(parity, reduce_and, reduce_or, parity, reduce_and, reduce_or)

; CHECK-LABEL: module Foo(
; CHECK:      wire [[PARITY:.+]] = ^x;
; CHECK-NOT:  ^x
; CHECK:      wire [[REDAND:.+]] = &y;
; CHECK-NOT:  &y
; CHECK:      wire [[REDOR:.+]] = |y;
; CHECK-NOT:  |y
; CHECK:      assign out = {[[PARITY]], [[REDAND]], [[REDOR]], [[PARITY]], [[REDAND]], [[REDOR]]};
; CHECK:      endmodule
