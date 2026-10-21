; RUN: firtool %s --disable-opt --verilog | FileCheck %s

; Check that the circt_sizeof intrinsic is properly evaluated for aggregate 
; types, even when optimizations are disabled.

; CHECK-LABEL: module Top(
; CHECK:          output [31:0] size
; CHECK:          assign size = 32'h0;

FIRRTL version 4.0.0
circuit Top:
  public module Top:
    output size: UInt<32>
    wire empty: {nested: {nested2: {nested3: UInt<1>[0]}} }
    invalidate empty
    node sizeNode = intrinsic(circt_sizeof : UInt<32>, empty)
    connect size, sizeNode
