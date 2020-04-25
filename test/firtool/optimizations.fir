; RUN: firtool %s --format=fir              | spt-opt | FileCheck %s --check-prefix=OPT
; RUN: firtool %s --format=fir -disable-opt | spt-opt | FileCheck %s --check-prefix=NOOPT

circuit optimizations :
  module test_cse :
    input a: UInt<4>
    output b: UInt<4>
    b <= add(a, a)
    b <= add(a, a)

; Only one add.
; OPT: %0 = firrtl.add %a, %a
; OPT: firrtl.connect %b, %0
; OPT: firrtl.connect %b, %0

; Both adds persist.
; NOOPT: %0 = firrtl.add %a, %a
; NOOPT: firrtl.connect %b, %0 
; NOOPT: %1 = firrtl.add %a, %a
; NOOPT: firrtl.connect %b, %1 

