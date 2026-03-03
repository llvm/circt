; RUN: firtool --btor2 %s | FileCheck %s
FIRRTL version 4.0.0

circuit Counter :
  public module Counter :
    input clock : Clock
    input reset : UInt<1>

    regreset count : UInt<32>, clock, reset, UInt<32>(0h0)
    node _T = eq(count, UInt<5>(0h16))
    when _T :
      connect count, UInt<1>(0h0)
    node _T_1 = neq(count, UInt<5>(0h16))
    when _T_1 :
      node _count_T = add(count, UInt<1>(0h1))
      node _count_T_1 = tail(_count_T, 1)
      connect count, _count_T_1
    node _T_2 = neq(count, UInt<4>(0ha))
    node hbr = intrinsic(circt_has_been_reset : UInt<1>, clock, reset)
    node ltl_clock = intrinsic(circt_ltl_clock : UInt<1>, _T_2, clock)
    intrinsic(circt_verif_assert, ltl_clock, hbr)

; CHECK:  1 sort bitvec 1
; CHECK:  2 input 1 clock
; CHECK:  3 input 1 reset
; CHECK:  4 sort bitvec 32
; CHECK:  5 state 4 count
; CHECK:  6 constd 1 0
; CHECK:  7 state 1 hbr
; CHECK:  8 init 1 7 6
; CHECK:  9 constd 4 1
; CHECK:  10 constd 4 10
; CHECK:  11 constd 4 22
; CHECK:  12 constd 4 0
; CHECK:  13 eq 1 5 11
; CHECK:  14 add 4 5 9
; CHECK:  15 ite 4 13 12 14
; CHECK:  16 neq 1 5 10
; CHECK:  17 constd 1 -1
; CHECK:  18 or 1 3 7
; CHECK:  19 xor 1 3 17
; CHECK:  20 and 1 7 19
; CHECK:  21 implies 1 20 16
; CHECK:  22 not 1 21
; CHECK:  23 bad 22
; CHECK:  24 ite 4 3 12 15
; CHECK:  25 next 4 5 24
; CHECK:  26 next 1 7 18
