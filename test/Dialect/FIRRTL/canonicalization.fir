; RUN: circt-translate -import-firrtl %s | circt-opt -canonicalize | FileCheck %s

; https://github.com/llvm/circt/issues/847
; seed: 12
circuit top_mod :
  ; CHECK-LABEL: firrtl.module @top_mod
  module top_mod :
    input clock: Clock
    input reset: UInt<1>
    input inp_h: UInt<0>
    wire tmp10: SInt<0>
    wire tmp25: SInt<29>
    wire tmp28: UInt<1>
    tmp10 <= asSInt(inp_h)
    tmp25 <= SInt<29>(-1)
    tmp28 <= geq(tmp10, tmp25)
    ; CHECK-NEXT: %tmp10 = firrtl.wire
    ; CHECK-NEXT: %0 = firrtl.asSInt %inp_h
    ; CHECK-NEXT: firrtl.connect %tmp10, %0
