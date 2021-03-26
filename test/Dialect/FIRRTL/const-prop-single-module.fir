; RUN: circt-translate --import-firrtl %s | circt-opt --canonicalize | FileCheck %s

; The following tests are derived from `ConstantPropagationSingleModule` in [1].
; [1]: https://github.com/chipsalliance/firrtl/blob/master/src/test/scala/firrtlTests/ConstantPropagationTests.scala

circuit ConstantPropagationSingleModule :
  module ConstantPropagationSingleModule:


; The rule x >= 0 should always be true if x is a UInt
  module Top01 :
    input x : UInt<5>
    output y : UInt<1>
    y <= geq(x, UInt(0))
; CHECK-LABEL: firrtl.module @Top01
; CHECK-NEXT: %[[K:.+]] = firrtl.constant(true)
; CHECK-NEXT: firrtl.connect %y, %[[K]]


; The rule x < 0 should never be true if x is a UInt
  module Top02 :
    input x : UInt<5>
    output y : UInt<1>
    y <= lt(x, UInt(0))
; CHECK-LABEL: firrtl.module @Top02
; CHECK-NEXT: %[[K:.+]] = firrtl.constant(false)
; CHECK-NEXT: firrtl.connect %y, %[[K]]


; The rule 0 <= x should always be true if x is a UInt
  module Top03 :
    input x : UInt<5>
    output y : UInt<1>
    y <= leq(UInt(0), x)
; CHECK-LABEL: firrtl.module @Top03
; CHECK-NEXT: %[[K:.+]] = firrtl.constant(true)
; CHECK-NEXT: firrtl.connect %y, %[[K]]


; The rule 0 > x should never be true if x is a UInt
  module Top04 :
    input x : UInt<5>
    output y : UInt<1>
    y <= gt(UInt(0), x)
; CHECK-LABEL: firrtl.module @Top04
; CHECK-NEXT: %[[K:.+]] = firrtl.constant(false)
; CHECK-NEXT: firrtl.connect %y, %[[K]]


; The rule 1 < 3 should always be true
  module Top05 :
    output y : UInt<1>
    y <= lt(UInt(1), UInt(3))
; CHECK-LABEL: firrtl.module @Top05
; CHECK-NEXT: %[[K:.+]] = firrtl.constant(true)
; CHECK-NEXT: firrtl.connect %y, %[[K]]


; The rule 10 == 10 should always be true
  module Top06 :
    output y : UInt<1>
    y <= eq(UInt(10), UInt(10))
; CHECK-LABEL: firrtl.module @Top06
; CHECK-NEXT: %[[K:.+]] = firrtl.constant(true)
; CHECK-NEXT: firrtl.connect %y, %[[K]]


; The rule x == z should not be true even if they have the same number of bits
  module Top07 :
    input x : UInt<3>
    input z : UInt<3>
    output y : UInt<1>
    y <= eq(x,z)
; CHECK-LABEL: firrtl.module @Top07
; CHECK-NEXT: %[[K:.+]] = firrtl.eq %x, %z
; CHECK-NEXT: firrtl.connect %y, %[[K]]


; The rule 10 != 10 should always be false
  module Top08 :
    output y : UInt<1>
    y <= neq(UInt(10), UInt(10))
; CHECK-LABEL: firrtl.module @Top08
; CHECK-NEXT: %[[K:.+]] = firrtl.constant(false)
; CHECK-NEXT: firrtl.connect %y, %[[K]]


; The rule 1 >= 3 should always be false
  module Top09 :
    output y : UInt<1>
    y <= geq(UInt(1), UInt(3))
; CHECK-LABEL: firrtl.module @Top09
; CHECK-NEXT: %[[K:.+]] = firrtl.constant(false)
; CHECK-NEXT: firrtl.connect %y, %[[K]]
