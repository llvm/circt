; RUN: firtool %s --disable-opt --verilog | FileCheck %s

; Check that the circt_sizeof intrinsic is properly evaluated, even when
; optimizations are disabled.

; CHECK-LABEL: module Top(
; CHECK:          output [31:0] size
; CHECK:          assign size = 32'h2A;

FIRRTL version 4.0.0
circuit Top :
  public module Top :
    output size: UInt<32>
    wire a: UInt<42>
    invalidate a
    node sizeNode = intrinsic(circt_sizeof : UInt<32>, a)
    connect size, sizeNode