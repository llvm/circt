; RUN: firtool --btor2 %s | FileCheck %s

FIRRTL version 4.0.0
circuit Counter :
  public module Counter :
    input clock : Clock
    input reset : UInt<1>
    input en : UInt<1>

    regreset count : UInt<32>, clock, reset, UInt<32>(0h0)

    when and(eq(count, UInt<32>(22)), en) :
      connect count, UInt<1>(0h0)

    when and(neq(count, UInt<32>(22)), en) :
      connect count, tail(add(count, UInt<1>(0h1)), 1)

    assert(clock, neq(count, UInt<4>(0ha)), en, "Counter reached 10!")

; CHECK:  1 sort bitvec 1
; CHECK:  2 input 1 clock
; CHECK:  3 input 1 reset
; CHECK:  4 input 1 en
; CHECK:  5 sort bitvec 32
; CHECK:  6 state 5 count
; CHECK:  7 constd 5 1
; CHECK:  8 constd 5 10
; CHECK:  9 constd 5 22
; CHECK:  10 constd 5 0
; CHECK:  11 eq 1 6 9
; CHECK:  12 and 1 11 4
; CHECK:  13 ite 5 12 10 6
; CHECK:  14 neq 1 6 9
; CHECK:  15 and 1 14 4
; CHECK:  16 add 5 6 7
; CHECK:  17 ite 5 15 16 13
; CHECK:  18 neq 1 6 8
; CHECK:  19 implies 1 4 18
; CHECK:  20 not 1 19
; CHECK:  21 bad 20
; CHECK:  22 ite 5 3 10 17
; CHECK:  23 next 5 6 22
