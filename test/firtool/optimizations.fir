; RUN: firtool %s --format=fir              | circt-opt | FileCheck %s --check-prefix=OPT
; RUN: firtool %s --format=fir -disable-opt | circt-opt | FileCheck %s --check-prefix=NOOPT

circuit test_cse :
  module test_cse :
    input a: UInt<4>
    output b: UInt<5>
    output c: UInt<5>
    output d: UInt<5>
    output e: UInt<5>
    b <= add(a, a)
    c <= add(a, a)
    d <= and(a, UInt<4>(0))
    e <= and(UInt<4>(3), UInt<4>(1))

; OPT-DAG: %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
; OPT-DAG: %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>

; Only one add.
; OPT: %0 = firrtl.add %a, %a
; OPT: firrtl.connect %b, %0
; OPT: firrtl.connect %c, %0

; Connect with zero and one directly.
; OPT: firrtl.connect %d, %c0_ui4
; OPT: firrtl.connect %e, %c1_ui4

; Both adds persist.
; NOOPT: %0 = firrtl.add %a, %a
; NOOPT: firrtl.connect %b, %0
; NOOPT: %1 = firrtl.add %a, %a
; NOOPT: firrtl.connect %c, %1

; Ands persist.
; NOOPT: %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
; NOOPT: firrtl.and %a, %c0_ui4
; NOOPT: firrtl.connect %d,

; NOOPT-DAG: %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
; NOOPT-DAG: %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
; NOOPT: firrtl.and %c3_ui4, %c1_ui4
; NOOPT: firrtl.connect %e,
