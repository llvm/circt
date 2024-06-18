; RUN: firtool --btor2 %s | FileCheck %s

FIRRTL version 4.0.0
circuit Counter :
  public module Counter :
    input clock : Clock
    input reset : UInt<1>
    input en : UInt<1>

    reg count : UInt<32>, clock with :
      reset => (reset, UInt<32>(0h0))

    when and(eq(count, UInt<32>(22)), en) :
      count <= UInt<1>(0h0)

    when and(neq(count, UInt<32>(22)), en) :
      count <= tail(add(count, UInt<1>(0h1)), 1)

    assert(clock, neq(count, UInt<4>(0ha)), en, "Counter reached 10!")

; CHECK:  1 sort bitvec 1
; CHECK:  2 input 1 reset
; CHECK:  3 input 1 en
; CHECK:  4 sort bitvec 32
; CHECK:  5 state 4 count
; CHECK:  6 constd 4 1
; CHECK:  7 constd 4 10
; CHECK:  8 constd 4 22
; CHECK:  9 constd 4 0
; CHECK:  10 eq 1 5 8
; CHECK:  11 and 1 10 3
; CHECK:  12 ite 4 11 9 5
; CHECK:  13 neq 1 5 8
; CHECK:  14 and 1 13 3
; CHECK:  15 add 4 5 6
; CHECK:  16 ite 4 14 15 12
; CHECK:  17 neq 1 5 7
; CHECK:  18 implies 1 3 17
; CHECK:  19 not 1 18
; CHECK:  20 bad 19
; CHECK:  21 ite 4 2 9 16
; CHECK:  22 next 4 5 21

