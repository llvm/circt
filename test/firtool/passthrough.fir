; For now, default to not enabling HW signal hoisting.
; RUN: firtool %s | FileCheck %s --check-prefix=DISABLE
; RUN: firtool -disable-hoisting-hw-passthrough=false %s | FileCheck %s --implicit-check-not module
; RUN: firtool -disable-opt %s | FileCheck %s --check-prefix=DISABLE
; RUN: firtool -disable-hoisting-hw-passthrough=true %s | FileCheck %s --check-prefix=DISABLE

; Check passthrough modules are elided.
; CHECK:       module HWMultiLevel
; CHECK:       );
; CHECK-EMPTY:
; CHECK-NEXT:    assign y = x & x2;
; CHECK-NEXT:  endmodule

; Check -disable-opt prevents this.
; DISABLE: module Child
; DISABLE: module Mid
; DISABLE: module HWMultiLevel
circuit HWMultiLevel:
  module Child:
    input x : UInt<5>
    output y : UInt<5>

    y <= x
  module Mid:
    input x : UInt<5>
    output y : UInt<5>

    inst c of Child
    c.x <= x
    y <= c.y

  module HWMultiLevel:
    input x : UInt<5>
    input x2 : UInt<5>
    output y : UInt<5>

    inst m of Mid
    inst m2 of Mid

    m.x <= x
    m2.x <= x2

    y <= and(m.y, m2.y)

