; RUN: firtool -split-input-file -ir-fir -lower-types=0 -imconstprop=0 %s | FileCheck %s
; XFAIL: *
; Tests extracted from:
;   - test/scala/firrtlTests/transforms/DedupTests.scala

; "The module A and A_ should be deduped even with different port names and info,
; and annotations should be remapped".
;
; This fails because the annotation shows up on an instance port and currently
; instance ports are not handled correctly.  This will be fixed once the "Lower
; Annotations" pass is load bearing and annotations will be canonicalized from
; on instance ports to as NLAs on module ports.
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top : %[[
  {
    "class": "circt.test",
     "target": "Top.Top.a2.y",
     "pin": "pin"
  }
]]
  ; CHECK: firrtl.module @Top
  module Top :
    output out: UInt<1>
    ; CHECK: inst a1 of A
    ; CHECK-NOT: SourceAnnotation
    ; CHECK: inst a2 of A
    ; CHECK-SAME: [{class = "circt.test", pin = "pin"}]
    inst a1 of A
    inst a2 of A_
    out <= and(a1.x, a2.y)
  module A : @[yy 2:2]
    output x: UInt<1> @[yy 2:2]
    x <= UInt(1)
  module A_ : @[xx 1:1]
    output y: UInt<1> @[xx 1:1]
    y <= UInt(1)

; CHECK-NOT: firrtl.module @A_


; // -----
; "The module A and A_ should dedup with different annotation targets."
;
; This test is checking that a "DontTouchAnnotation" has the same behavior as
; any other annotation due to deduplication.  Presently, this is not the case
; because a "DontTouchAnnotation" is promoted to a symbol.
;
; See: https://github.com/llvm/circt/issues/2531
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top : %[[
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"Top.A.b"
  }
]]
  ; CHECK: firrtl.module @Top
  ; CHECK-NEXT: firrtl.instance a1 sym @[[a1Sym:[_a-zA-Z0-9]+]] {annotations = [{circt.nonlocal = @[[nlaSym:[_a-zA-Z0-9]+]], class = "circt.nonlocal"}]} @A(
  ; CHECK-NEXT: firrtl.instance a2 @A(
  module Top :
    inst a1 of A
    inst a2 of A_
  ; CHECK: hw.hierpath private @[[nlaSym]] [@Top::@[[a1Sym]], @A::@[[bSym:[_a-zA-Z0-9]+]]]
  module A :
    output x: UInt<1>
    ; CHECk-NEXT: firrtl.wire sym @[[bSym]] {annotations = [{circt.nonlocal = @[[nlaSym]], class = "firrtl.transforms.DontTouchAnnotation"}]}
    wire b: UInt<1>
    b is invalid
    x <= b
  module A_ :
    output x: UInt<1>
    wire b: UInt<1>
    b is invalid
    x <= b

; CHECK-NOT: firrtl.module @A_

; // -----
; "The module A and A_ should dedup with the same annotation targets."
;
; This is the same situation as the previous test, except that two
; "DontTouchAnnotation"s apply to the same deduplicated wire.  It is arguable
; that generating a symbol here is fine.  However, this is different from the
; Scala FIRRTL Compiler and we should make a decision on what to do with this.
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top : %[[
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"Top.A.b"
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Top|A_>b"
  }
]]
  ; CHECK: firrtl.module @Top
  ; CHECK-NEXT: firrtl.instance a1 sym @[[a1Sym:[_a-zA-Z0-9]+]] {annotations = [{circt.nonlocal = @[[nlaSym1:[_a-zA-Z0-9]+]], class = "circt.nonlocal"}]} @A(
  ; CHECK-NEXT: firrtl.instance a2 sym @[[a2Sym:[_a-zA-Z0-9]+]] {annotations = [{circt.nonlocal = @[[nlaSym2:[_a-zA-Z0-9]+]], class = "circt.nonlocal"}]} @A(
  module Top :
    inst a1 of A
    inst a2 of A_
  ; CHECK: hw.hierpath private @[[nlaSym1]] [@Top::@[[a1Sym1]], @A::@[[bSym:[_a-zA-Z0-9]+]]]
  ; CHECK: hw.hierpath private @[[nlaSym2]] [@Top::@[[a1Sym2]], @A::@[[bSym]]]
  module A :
    output x: UInt<1>
    ; CHECk-NEXT: firrtl.wire sym @[[bSym]]
    ; CHECK-SAME: {circt.nonlocal = @[[nlaSym1]], class = "firrtl.transform.DontTouchAnnotation"}
    ; CHECK-SAME: {circt.nonlocal = @[[nlaSym2]], class = "firrtl.transform.DontTouchAnnotation"}
    wire b: UInt<1>
    b is invalid
    x <= b
  module A_ :
    output x: UInt<1>
    wire b: UInt<1>
    b is invalid
    x <= b

; CHECK-NOT: firrtl.module @A_

; // -----
; "modules should not be deduped if the NoCircuitDedupAnnotation is supplied"
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top: %[[
  {
    "class":"firrtl.transforms.NoCircuitDedupAnnotation"
  }
]]
  ; CHECK: firrtl.module @A(
  module A:
    input a: UInt<1>
    output b: UInt<1>
    b <= a
  ; CHECK: firrtl.module @A_(
  module A_:
    input a: UInt<1>
    output b: UInt<1>
    b <= a
  ; CHECK: firrtl.module @Top(
  module Top:
    input a: UInt<1>
    output b: UInt<1>

    ; CHECK-NEXT: firrtl.instance a1 @A(
    ; CHECK-NEXT: firrtl.instance a2 @A_(
    inst a1 of A
    inst a2 of A_
    a1.a <= a
    a2.a <= a
    b <= xor(a1.b, a2.b)
