; RUN: firtool --btor2 %s | FileCheck %s

FIRRTL version 3.2.0
circuit Counter :
  intmodule HasBeenResetIntrinsic :
    input clock : Clock
    input reset : Reset
    output out : UInt<1>
    intrinsic = circt_has_been_reset

  intmodule LTLDisableIntrinsic :
    input in : UInt<1>
    input condition : UInt<1>
    output out : UInt<1>
    intrinsic = circt_ltl_disable

  intmodule LTLClockIntrinsic :
    input in : UInt<1>
    input clock : Clock
    output out : UInt<1>
    intrinsic = circt_ltl_clock

  intmodule VerifAssertIntrinsic :
    input property : UInt<1>
    intrinsic = circt_verif_assert

  module Counter :
    input clock : Clock
    input reset : UInt<1>

    regreset count : UInt<32>, clock, reset, UInt<32>(0h0) @[src/test/scala/ltl/LTLSpec.scala 31:28]
    node _T = eq(count, UInt<5>(0h16)) @[src/test/scala/ltl/LTLSpec.scala 32:20]
    when _T : @[src/test/scala/ltl/LTLSpec.scala 32:30]
      connect count, UInt<1>(0h0) @[src/test/scala/ltl/LTLSpec.scala 32:38]
    node _T_1 = neq(count, UInt<5>(0h16)) @[src/test/scala/ltl/LTLSpec.scala 33:20]
    when _T_1 : @[src/test/scala/ltl/LTLSpec.scala 33:30]
      node _count_T = add(count, UInt<1>(0h1)) @[src/test/scala/ltl/LTLSpec.scala 33:47]
      node _count_T_1 = tail(_count_T, 1) @[src/test/scala/ltl/LTLSpec.scala 33:47]
      connect count, _count_T_1 @[src/test/scala/ltl/LTLSpec.scala 33:38]
    node _T_2 = neq(count, UInt<4>(0ha)) @[src/test/scala/ltl/LTLSpec.scala 34:30]
    inst HasBeenResetIntrinsic of HasBeenResetIntrinsic @[src/main/scala/chisel3/ltl/LTL.scala 357:39]
    connect HasBeenResetIntrinsic.clock, clock @[src/main/scala/chisel3/ltl/LTL.scala 357:39]
    connect HasBeenResetIntrinsic.reset, reset @[src/main/scala/chisel3/ltl/LTL.scala 357:39]
    node disable = eq(HasBeenResetIntrinsic.out, UInt<1>(0h0)) @[src/main/scala/chisel3/ltl/LTL.scala 357:39]
    inst ltl_disable of LTLDisableIntrinsic @[src/main/scala/chisel3/ltl/LTL.scala 329:34]
    connect ltl_disable.in, _T_2 @[src/main/scala/chisel3/ltl/LTL.scala 330:20]
    connect ltl_disable.condition, disable @[src/main/scala/chisel3/ltl/LTL.scala 331:27]
    inst ltl_clock of LTLClockIntrinsic @[src/main/scala/chisel3/ltl/LTL.scala 318:32]
    connect ltl_clock.in, ltl_disable.out @[src/main/scala/chisel3/ltl/LTL.scala 319:18]
    connect ltl_clock.clock, clock @[src/main/scala/chisel3/ltl/LTL.scala 320:21]
    inst verif of VerifAssertIntrinsic @[src/main/scala/chisel3/ltl/LTL.scala 375:59]
    connect verif.property, ltl_clock.out @[src/main/scala/chisel3/ltl/LTL.scala 363:20]

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
; CHECK:  11 constd 1 -1
; CHECK:  12 constd 3 0
; CHECK:  13 eq 1 4 10
; CHECK:  14 add 3 4 8
; CHECK:  15 ite 3 13 12 14
; CHECK:  16 neq 1 4 9
; CHECK:  17 constd 1 -1
; CHECK:  18 or 1 2 6
; CHECK:  19 xor 1 2 17
; CHECK:  20 and 1 6 19
; CHECK:  21 xor 1 20 11
; CHECK:  22 or 1 21 16
; CHECK:  23 not 1 22
; CHECK:  24 bad 23
; CHECK:  25 ite 3 2 12 15
; CHECK:  26 next 3 4 25
; CHECK:  28 next 1 6 27
