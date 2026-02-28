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
; CHECK:  2 input 1 reset
; CHECK:  3 sort bitvec 32
; CHECK:  4 state 3 count
; CHECK:  5 constd 1 0
; CHECK:  6 state 1 hbr
; CHECK:  7 init 1 6 5
; CHECK:  8 constd 3 1
; CHECK:  9 constd 3 10
; CHECK:  10 constd 3 22
; CHECK:  11 constd 3 0
; CHECK:  12 eq 1 4 10
; CHECK:  13 add 3 4 8
; CHECK:  14 ite 3 12 11 13
; CHECK:  15 neq 1 4 9
; CHECK:  16 constd 1 -1
; CHECK:  17 or 1 2 6
; CHECK:  18 xor 1 2 16
; CHECK:  19 and 1 6 18
; CHECK:  20 implies 1 19 15
; CHECK:  21 not 1 20
; CHECK:  22 bad 21
; CHECK:  23 ite 3 2 11 14
; CHECK:  24 next 3 4 23
; CHECK:  25 next 1 6 17
