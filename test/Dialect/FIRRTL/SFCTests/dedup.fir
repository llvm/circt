; RUN: firtool -dedup -split-input-file -ir-fir -lower-types=0 -imconstprop=0 -remove-unused-ports=0 %s | FileCheck %s
; Tests extracted from:
;   - test/scala/firrtlTests/transforms/DedupTests.scala
; The following tests are not included:
;   - "The module A and A_ should be deduped with the same annotation targets
;     when there are a lot" because this is checking an internal bug in SFC.
;   - "The module A and A_ should be dedup with same annotations with same
;     multi-targets" because multi-target annotations are not used in CIRCT.
;     Instead multiple annotations are scattered into the circuit.  An
;     approximation of multi-target annotations (which uses multiple annotations
;     with the same payload) is tested in a later test.
;   - "The module A and A_ should be deduped with same annotations with same
;     multi-targets, that share roots" for the same reason as above.

; "The module A should be deduped"
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  ; CHECK: firrtl.module @Top
  module Top :
    ; CHECK-COUNT-2: firrtl.instance {{a(1|2)}} @A(
    inst a1 of A
    inst a2 of A_
  module A :
    output x: UInt<1>
    x <= UInt(1)
  module A_ :
    output x: UInt<1>
    x <= UInt(1)

; CHECK-NOT: firrtl.module private @A_

; // -----
; "The module A and B should be deduped"
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  ; CHECK: firrtl.module @Top
  module Top :
    ; CHECK-NEXT-COUNT-2: firrtl.instance {{a(1|2)}} @A(
    inst a1 of A
    inst a2 of A_
  ; CHECK: firrtl.module private @A
  module A :
    output x: UInt<1>
    ; CHECK: firrtl.instance b @B(
    inst b of B
    x <= b.x
  module A_ :
    output x: UInt<1>
    inst b of B_
    x <= b.x
  module B :
    output x: UInt<1>
    x <= UInt(1)
  module B_ :
    output x: UInt<1>
    x <= UInt(1)

; CHECK-NOT: firrtl.module private @A_
; CHECK-NOT: firrtl.module private @B_

; // -----
; "The module A and B with comments should be deduped"
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  ; CHECK: firrtl.module @Top
  module Top :
    ; CHECK: firrtl.instance a1 @A(
    ; CHECK: firrtl.instance a2 @A(
    inst a1 of A
    inst a2 of A_
  module A : @[yy 2:2]
    output x: UInt<1> @[yy 2:2]
    inst b of B @[yy 2:2]
    x <= b.x @[yy 2:2]
  module A_ : @[xx 1:1]
    output x: UInt<1> @[xx 1:1]
    inst b of B_ @[xx 1:1]
    x <= b.x @[xx 1:1]
  module B :
    output x: UInt<1>
    x <= UInt(1)
  module B_ :
    output x: UInt<1>
    x <= UInt(1)

; CHECK-NOT: firrtl.module private @A_
; CHECK-NOT: firrtl.module private @B_

; // -----
; "A_ but not A should be deduped if not annotated"
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  ; CHECK: firrtl.module @Top
  module Top :
    ; CHECK-NEXT-COUNT-2: firrtl.instance {{a(1|2)}} @A(
    inst a1 of A
    inst a2 of A_
  module A : @[yy 2:2]
    output x: UInt<1> @[yy 2:2]
    x <= UInt(1)
  module A_ : @[xx 1:1]
    output x: UInt<1> @[xx 1:1]
    x <= UInt(1)

; CHECK-NOT: firrtl.module private @A_

; // -----
; "Extmodules with the same defname and parameters should dedup"
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  ; CHECK: firrtl.module @Top
  module Top :
    output out: UInt<1>
    ; CHECK-NEXT-COUNT-2: firrtl.instance {{a(1|2)}} @A(
    inst a1 of A
    inst a2 of A_
    out <= and(a1.x, a2.y)
  ; CHECK: firrtl.module private @A
  module A : @[yy 2:2]
    output x: UInt<1> @[yy 2:2]
    ; CHECK-NEXT: firrtl.instance b @B(
    inst b of B
    x <= b.u
  module A_ : @[xx 1:1]
    output y: UInt<1> @[xx 1:1]
    inst c of C
    y <= c.u
  extmodule B : @[aa 3:3]
    output u : UInt<1> @[aa 4:4]
    defname = BB
    parameter N = 0
  extmodule C : @[bb 5:5]
    output u : UInt<1> @[bb 6:6]
    defname = BB
    parameter N = 0

; CHECK-NOT: firrtl.module private @A_
; CHECK-NOT: firrtl.extmodule private @C

; // -----
; "Extmodules with different defname should NOT dedup"
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  ; CHECK: firrtl.module @Top
  module Top :
    output out: UInt<1>
    ; CHECK-NEXT: firrtl.instance a1 @A(
    ; CHECK-NEXT: firrtl.instance a2 @A_(
    inst a1 of A
    inst a2 of A_
    out <= and(a1.x, a2.y)
  module A : @[yy 2:2]
    output x: UInt<1> @[yy 2:2]
    inst b of B
    x <= b.u
  ; CHECK: firrtl.module private @A_
  module A_ : @[xx 1:1]
    output y: UInt<1> @[xx 1:1]
    inst c of C
    y <= c.u
  ; CHECK: firrtl.extmodule private @B
  extmodule B : @[aa 3:3]
    output u : UInt<1> @[aa 4:4]
    defname = BB
    parameter N = 0
  ; CHECK: firrtl.extmodule private @C
  extmodule C : @[bb 5:5]
    output u : UInt<1> @[bb 6:6]
    defname = CD
    parameter N = 0

; // -----
; "Extmodules with different parameters should NOT dedup"
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  ; CHECK: firrtl.module @Top
  module Top :
    output out: UInt<1>
    ; CHECK-NEXT: firrtl.instance a1 @A(
    ; CHECK-NEXT: firrtl.instance a2 @A_(
    inst a1 of A
    inst a2 of A_
    out <= and(a1.x, a2.y)
  module A : @[yy 2:2]
    output x: UInt<1> @[yy 2:2]
    inst b of B
    x <= b.u
  ; CHECK: firrtl.module private @A_
  module A_ : @[xx 1:1]
    output y: UInt<1> @[xx 1:1]
    inst c of C
    y <= c.u
  ; CHECK: firrtl.extmodule private @B
  extmodule B : @[aa 3:3]
    output u : UInt<1> @[aa 4:4]
    defname = BB
    parameter N = 0
  ; CHECK: firrtl.extmodule private @C
  extmodule C : @[bb 5:5]
    output u : UInt<1> @[bb 6:6]
    defname = BB
    parameter N = 1

; // -----
; "Modules with aggregate ports that are (partial)? connected should NOT dedup if
; their port names differ".
;
; Since the MFC support expanding connects and partial connects when the port
; names differ, this now testing that the modules succesfully dedup.
;
; CHECK-LABEL: firrtl.circuit "FooAndBarModule"
circuit FooAndBarModule :
  ; CHECK: firrtl.module private @FooModule
  module FooModule :
    output io : {flip foo : UInt<1>, fuzz : UInt<1>}
    io.fuzz <= io.foo
  ; CHECK-NOT: firrtl.module private @BarModule
  module BarModule :
    output io : {flip bar : UInt<1>, buzz : UInt<1>}
    io.buzz <= io.bar
  ; CHECK: firrtl.module @FooAndBarModule
  module FooAndBarModule :
    output io : {foo : {flip foo : UInt<1>, fuzz : UInt<1>}, bar : {flip bar : UInt<1>, buzz : UInt<1>}}
    ; CHECK: firrtl.instance foo @FooModule
    ; CHECK: firrtl.instance bar @FooModule
    inst foo of FooModule
    inst bar of BarModule
    io.foo <- foo.io
    io.bar <= bar.io

; // -----
; "Modules with aggregate ports that are (partial)? connected should dedup if
; their port names are the same".
;
; CHECK-LABEL: firrtl.circuit "FooAndBarModule"
circuit FooAndBarModule :
  ; CHECK: firrtl.module private @FooModule
  module FooModule :
    output io : {flip foo : UInt<1>, fuzz : UInt<1>}
    io.fuzz <= io.foo
  ; CHECK-NOT: firrtl.module private @BarModule
  module BarModule :
    output io : {flip foo : UInt<1>, fuzz : UInt<1>}
    io.fuzz <= io.foo
  ; CHECK: firrtl.module @FooAndBarModule
  module FooAndBarModule :
    output io : {foo : {flip foo : UInt<1>, fuzz : UInt<1>}, bar : {flip foo : UInt<1>, fuzz : UInt<1>}}
    ; CHECK-NEXT-COUNT-2: firrtl.instance {{(foo|bar)}} @FooModule
    inst foo of FooModule
    inst bar of BarModule
    io.foo <= foo.io
    io.bar <- bar.io

; // -----
; "The module A and B should be deduped with the first module in order".
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  ; CHECK: firrtl.module @Top
  module Top :
    ; CHECK-NEXT-COUNT-2: firrtl.instance {{a(1|2)}} @A(
    inst a1 of A
    inst a2 of A_
  module A :
    output x: UInt<1>
    inst b of B
    x <= b.x
  module A_ :
    output x: UInt<1>
    inst b of B_
    x <= b.x
  module B :
    output x: UInt<1>
    x <= UInt(1)
  module B_ :
    output x: UInt<1>
    x <= UInt(1)

; CHECK-NOT: firrtl.module private @A_
; CHECK-NOT: firrtl.module private @B_

; // -----
; "The module A and A_ should be deduped with fields that sort of match".
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top :
  ; CHECK: firrtl.module @Top
  module Top :
    ; CHECK-NEXT-COUNT-2: firrtl.instance {{a(1|2)}} @A(
    inst a1 of A
    inst a2 of A_
  module A :
    output x: UInt<1>
    wire b: {c: UInt<1>}
    b is invalid
    x <= b.c
  module A_ :
    output x: UInt<1>
    wire b: {b: UInt<1>}
    b is invalid
    x <= b.b

; CHECK-NOT: firrtl.module private @A_

; // -----
; "The module A and A_ should dedup with different annotation targets".
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top : %[[
  {
    "class":"hello",
    "target":"Top.A.b"
  }
]]
  ; CHECK: firrtl.nla @[[nlaSym:[_a-zA-Z0-9]+]] [@Top::@[[a1Sym:[_a-zA-Z0-9]+]], @A::@[[bSym:[_a-zA-Z0-9]+]]]
  ; CHECK: firrtl.module @Top
  ; CHECK-NEXT: firrtl.instance a1 sym @[[a1Sym]] @A(
  ; CHECK-NEXT: firrtl.instance a2 @A(
  module Top :
    inst a1 of A
    inst a2 of A_
  module A :
    output x: UInt<1>
    ; CHECk-NEXT: firrtl.wire sym @[[bSym]] {annotations = [{circt.nonlocal = @[[nlaSym]], class = "hello"}]}
    wire b: UInt<1>
    b is invalid
    x <= b
  module A_ :
    output x: UInt<1>
    wire b: UInt<1>
    b is invalid
    x <= b

; CHECK-NOT: firrtl.module private @A_

; // -----
; "The module A and A_ should dedup with the same annotation targets".
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top : %[[
  {
    "class":"hello",
    "target":"Top.A.b"
  },
  {
    "class":"world",
    "target":"~Top|A_>b"
  }
]]
  ; CHECK: firrtl.nla @[[nlaSym1:[_a-zA-Z0-9]+]] [@Top::@[[a1Sym:[_a-zA-Z0-9]+]], @A::@[[bSym:[_a-zA-Z0-9]+]]]
  ; CHECK: firrtl.nla @[[nlaSym2:[_a-zA-Z0-9]+]] [@Top::@[[a2Sym:[_a-zA-Z0-9]+]], @A::@[[bSym]]]
  ; CHECK: firrtl.module @Top
  ; CHECK-NEXT: firrtl.instance a1 sym @[[a1Sym]] @A(
  ; CHECK-NEXT: firrtl.instance a2 sym @[[a2Sym]] @A(
  module Top :
    inst a1 of A
    inst a2 of A_
  module A :
    output x: UInt<1>
    ; CHECk-NEXT: firrtl.wire sym @[[bSym]] {annotations = [{circt.nonlocal = @[[nlaSym1]], class = "world"}, {circt.nonlocal = @[[nlaSym2]], class = "hello"}]}
    wire b: UInt<1>
    b is invalid
    x <= b
  module A_ :
    output x: UInt<1>
    wire b: UInt<1>
    b is invalid
    x <= b

; CHECK-NOT: firrtl.module private @A_

; // -----
; "The deduping module A and A_ should rename internal signals that have
; different names".  This is checking that an NLA is not created even though
; structural deduplication changes the name of "A_>b" to "A>a".
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top : %[[
  {
    "class":"hello",
    "target":"~Top|A>a"
  },
  {
    "class":"hello",
    "target":"~Top|A_>b"
  }
]]
  ; CHECK-NOT: firrtl.nla
  module Top :
    inst a1 of A
    a1 is invalid
    inst a2 of A_
    a2 is invalid
  ; CHECK: firrtl.module private @A(
  module A :
    input x: UInt<1>
    output y: UInt<1>
    ; CHECK: firrtl.node {{.+}} {annotations = [{class = "hello"}]}
    node a = add(x, UInt(1))
    y <= add(a, a)
  module A_ :
    input x: UInt<1>
    output y: UInt<1>
    node b = add(x, UInt(1))
    y <= add(b, b)

; CHECK-NOT: firrtl.module private @A_(

; // -----
; "main should not be deduped even if it's the last module"
;
; CHECK-LABEL: firrtl.circuit "main"
circuit main:
  module dupe:
    input in: UInt<8>
    output out: UInt<8>
    out <= in
  ; CHECK: firrtl.module @main
  module main:
    input in:  UInt<8>
    output out: UInt<8>
    out <= in

; // -----
; "The deduping module A and A_ should rename instances and signals that have different names"
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top : %[[
  {
    "class":"B",
    "target":"~Top|Top/a:A/b:B"
  },
  {
    "class":"B.foo",
    "target":"~Top|Top/a:A/b:B>foo"
  },
  {
    "class":"B_",
    "target":"~Top|Top/a_:A_/b_:B_"
  },
  {
    "class":"B_.bar",
    "target":"~Top|Top/a_:A_/b_:B_>bar"
  }
]]
  ; CHECK: firrtl.nla @[[nla_4:[_a-zA-Z0-9]+]] [@Top::@[[a_Sym:[_a-zA-Z0-9]+]], @A::@[[bSym:[_a-zA-Z0-9]+]], @B::@[[fooSym:[_a-zA-Z0-9]+]]]
  ; CHECK: firrtl.nla @[[nla_3:[_a-zA-Z0-9]+]] [@Top::@[[a_Sym]], @A::@[[bSym]], @B]
  ; CHECK: firrtl.nla @[[nla_2:[_a-zA-Z0-9]+]] [@Top::@[[aSym:[_a-zA-Z0-9]+]], @A::@[[bSym]], @B::@[[fooSym]]]
  ; CHECK: firrtl.nla @[[nla_1:[_a-zA-Z0-9]+]] [@Top::@[[aSym]], @A::@[[bSym]], @B]
  ; CHECK: firrtl.module @Top
  module Top :
    ; CHECK-NEXT: firrtl.instance a sym @[[aSym]] @A()
    inst a of A
    ; CHECK-NEXT: firrtl.instance a_ sym @[[a_Sym]] @A()
    inst a_ of A_
  ; CHECK: firrtl.module private @A
  module A :
    ; CHECK-NEXT: firrtl.instance b sym @[[bSym]]
    ; CHECK-SAME: B()
    inst b of B
  ; CHECK-NOT: firrtl.module private @A_
  module A_ :
    inst b_ of B_
  ; CHECK: firrtl.module private @B
  ; CHECK-SAME: {circt.nonlocal = @[[nla_3]], class = "B_"}
  ; CHECK-SAME: {circt.nonlocal = @[[nla_1]], class = "B"}
  module B :
    ; CHECK: firrtl.node sym @[[fooSym]]
    ; CHECK-SAME: {circt.nonlocal = @[[nla_4]], class = "B_.bar"}
    ; CHECK-SAME: {circt.nonlocal = @[[nla_2]], class = "B.foo"}
    node foo = UInt<1>(0)
  ; CHECK-NOT: firrtl.module private @B_
  module B_ :
    node bar = UInt<1>(0)

; // -----
;
; "The deduping module A and A_ should rename nested instances that have
; different names".  The original test used two multi-target annotations where
; each annotation targets both a module and a reference in the module.  This
; tests approximates this with two annotations with the same class that mimics
; how a multi-target annotation would be scattered.
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top : %[[
  {
    "class":"one",
    "target":"~Top|Top/a:A/b:B/c:C/d:D"
  },
  {
    "class":"one",
    "target":"~Top|Top/a:A/b:B/c:C/d:D>foo"
  },
  {
    "class":"two",
    "target":"~Top|Top/a_:A_/b_:B_/c_:C_/d_:D_"
  },
  {
    "class":"two",
    "target":"~Top|Top/a_:A_/b_:B_/c_:C_/d_:D_>bar"
  }
]]
  ; CHECK:        firrtl.nla @[[nla_4:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[a_Sym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @A::@[[bSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @B::@[[cSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @C::@[[dSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @D::@[[fooSym:[_a-zA-Z0-9]+]]]
  ; CHECK-NEXT:   firrtl.nla @[[nla_3:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[a_Sym]],
  ; CHECK-SAME:      @A::@[[bSym]],
  ; CHECK-SAME:      @B::@[[cSym]],
  ; CHECK-SAME:      @C::@[[dSym]],
  ; CHECK-SAME:      @D]
  ; CHECK-NEXT:   firrtl.nla @[[nla_2:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[aSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @A::@[[bSym]],
  ; CHECK-SAME:      @B::@[[cSym]],
  ; CHECK-SAME:      @C::@[[dSym]],
  ; CHECK-SAME:      @D::@[[fooSym]]]
  ; CHECK-NEXT:   firrtl.nla @[[nla_1:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[aSym]],
  ; CHECK-SAME:      @A::@[[bSym]],
  ; CHECK-SAME:      @B::@[[cSym]],
  ; CHECK-SAME:      @C::@[[dSym]],
  ; CHECK-SAME:      @D]
  ; CHECK-NEXT:   firrtl.module @Top
  module Top :
    ; CHECK-NEXT:   firrtl.instance a sym @[[aSym]]
    ; CHECK-SAME:     @A()
    inst a of A
    ; CHECK-NEXT:   firrtl.instance a_ sym @[[a_Sym]]
    ; CHECK-SAME:     @A()
    inst a_ of A_

  ; CHECK:        firrtl.module private @A
  module A :
    ; CHECK-NEXT:   firrtl.instance b sym @[[bSym]]
    ; CHECK-SAME:     @B()
    inst b of B

  ; CHECK:        firrtl.module private @B
  module B :
    ; CHECK-NEXT:   firrtl.instance c sym @[[cSym]]
    ; CHECK-SAME:     @C()
    inst c of C

  ; CHECK:        firrtl.module private @C
  module C :
    ; CHECK-NEXT:   firrtl.instance d sym @[[dSym]]
    ; CHECK-SAME:     @D()
    inst d of D

  ; CHECK:        firrtl.module private @D
  module D :
    ; CHECK:        firrtl.node
    node foo = UInt<1>(0)

  ; CHECK-NOT: firrtl.module private @A_
  ; CHECK-NOT: firrtl.module private @B_
  ; CHECK-NOT: firrtl.module private @C_
  ; CHECK-NOT: firrtl.module private @D_
  module A_ :
    inst b_ of B_
  module B_ :
    inst c_ of C_
  module C_ :
    inst d_ of D_
  module D_ :
    node bar = UInt<1>(0)

; // -----
;
; "Deduping modules with multiple instances should correctly rename
; instances".
;
; CHECK-LABEL: firrtl.circuit "Top"
circuit Top : %[[
  {
    "class":"nla1",
    "target":"~Top|Top/b:B"
  },
  {
    "class":"nla2",
    "target":"~Top|Top/b_:B_"
  },
  {
    "class":"nla3",
    "target":"~Top|Top/a1:A/b_:B_"
  },
  {
    "class":"nla4",
    "target":"~Top|Top/a2:A/b_:B_"
  },
  {
    "class":"nla5",
    "target":"~Top|Top/a1:A/b:B"
  },
  {
    "class":"nla6",
    "target":"~Top|Top/a2:A/b:B"
  },
  {
    "class":"nla7",
    "target":"~Top|Top/b:B/c:C"
  },
  {
    "class":"nla8",
    "target":"~Top|Top/b_:B_/c:C"
  },
  {
    "class":"nla9",
    "target":"~Top|Top/a1:A/b_:B_/c:C"
  },
  {
    "class":"nla10",
    "target":"~Top|Top/a2:A/b_:B_/c:C"
  },
  {
    "class":"nla11",
    "target":"~Top|Top/a1:A/b:B/c:C"
  },
  {
    "class":"nla12",
    "target":"~Top|Top/a2:A/b:B/c:C"
  }
]]
  ; CHECK:        firrtl.nla @[[nla_12:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[Topa2Sym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @A::@[[AbSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @B::@[[BcSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @C]
  ; CHECK:        firrtl.nla @[[nla_11:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[Topa1Sym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @A::@[[AbSym]],
  ; CHECK-SAME:      @B::@[[BcSym]],
  ; CHECK-SAME:      @C]
  ; CHECK:        firrtl.nla @[[nla_10:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[Topa2Sym]],
  ; CHECK-SAME:      @A::@[[Ab_Sym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @B::@[[BcSym]],
  ; CHECK-SAME:      @C]
  ; CHECK:        firrtl.nla @[[nla_9:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[Topa1Sym]],
  ; CHECK-SAME:      @A::@[[Ab_Sym]],
  ; CHECK-SAME:      @B::@[[BcSym]],
  ; CHECK-SAME:      @C]
  ; CHECK:        firrtl.nla @[[nla_8:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[Topb_Sym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @B::@[[BcSym]],
  ; CHECK-SAME:      @C]
  ; CHECK:        firrtl.nla @[[nla_7:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[TopbSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @B::@[[BcSym]],
  ; CHECK-SAME:      @C]
  ; CHECK:        firrtl.nla @[[nla_6:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[Topa2Sym]],
  ; CHECK-SAME:      @A::@[[AbSym]],
  ; CHECK-SAME:      @B]
  ; CHECK:        firrtl.nla @[[nla_5:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[Topa1Sym]],
  ; CHECK-SAME:      @A::@[[AbSym]],
  ; CHECK-SAME:      @B]
  ; CHECK:        firrtl.nla @[[nla_4:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[Topa2Sym]],
  ; CHECK-SAME:      @A::@[[Ab_Sym]],
  ; CHECK-SAME:      @B]
  ; CHECK:        firrtl.nla @[[nla_3:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[Topa1Sym]],
  ; CHECK-SAME:      @A::@[[Ab_Sym]],
  ; CHECK-SAME:      @B]
  ; CHECK:        firrtl.nla @[[nla_2:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[Topb_Sym]],
  ; CHECK-SAME:      @B]
  ; CHECK:        firrtl.nla @[[nla_1:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top::@[[TopbSym]],
  ; CHECK-SAME:      @B]
  ; CHECK-NEXT:   firrtl.module @Top
  module Top :
    ; CHECK-NEXT:   firrtl.instance b sym @[[TopbSym]]
    ; CHECK-SAME:     @B()
    inst b of B
    ; CHECK-NEXT:   firrtl.instance b_ sym @[[TopbSym]]
    ; CHECK-SAME:     @B()
    inst b_ of B_
    ; CHECK-NEXT:   firrtl.instance a1 sym @[[Topa1Sym]]
    ; CHECK-SAME:     @A()
    inst a1 of A
    ; CHECK-NEXT:   firrtl.instance a2 sym @[[Topa2Sym]]
    ; CHECK-SAME:     @A()
    inst a2 of A

  ; CHECK:        firrtl.module private @A()
  module A :
    ; CHECK-NEXT:   firrtl.instance b sym @[[AbSym]]
    ; CHECK-SAME:     @B()
    inst b of B
    ; CHECK-NEXT:   firrtl.instance b_ sym @[[Ab_Sym]]
    ; CHECK-SAME:     @B()
    inst b_ of B_

  ; CHECK:        firrtl.module private @B()
  ; CHECK-SAME:     [{circt.nonlocal = @[[nla_2]], class = "nla2"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_3]], class = "nla3"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_4]], class = "nla4"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_1]], class = "nla1"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_5]], class = "nla5"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_6]], class = "nla6"}]
  module B :
    ; CHECK-NEXT:   firrtl.instance c sym @[[BcSym]]
    ; CHECK-SAME:     @C()
    inst c of C

  ; CHECK:        firrtl.module private @C()
  ; CHECK-SAME:     [{circt.nonlocal = @[[nla_7]], class = "nla7"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_8]], class = "nla8"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_9]], class = "nla9"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_10]], class = "nla10"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_11]], class = "nla11"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_12]], class = "nla12"}]
  module C :
    skip

  ; CHECK-NOT: firrtl.module private @B_()
  module B_ :
    inst c of C

; // -----
;
; "dedup should properly rename target components after retyping".  This test is
; checking for a possible deduplication bug where there is a string name
; collision of the field of a merged aggregate.  This test is very specific to
; the implementation of the Scala FIRRTL Compiler's rename map (which relies on
; strings) and is unlikely to occur in CIRCT due to the use of field IDs.
;
; CHECK-LABEL: firrtl.circuit "top"
circuit top : %[[
  {
    "class":"nla1",
    "target":"~top|a>i.a"
  },
  {
    "class":"nla2",
    "target":"~top|b>q.a"
  }
]]
  ; CHECK:      firrtl.nla @[[nla_1:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:   [@top::@[[topaSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:    @a::@[[aiSym:[_a-zA-Z0-9]+]]
  ; CHECK:      firrtl.nla @[[nla_2:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:   [@top::@[[topbSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:    @a::@[[aiSym]]
  ; CHECK: firrtl.module @top
  module top:
    input ia: {z: {y: {x: UInt<1>}}, a: UInt<1>}
    input ib: {a: {b: {c: UInt<1>}}, z: UInt<1>}
    output oa: {z: {y: {x: UInt<1>}}, a: UInt<1>}
    output ob: {a: {b: {c: UInt<1>}}, z: UInt<1>}
    ; CHECK:      firrtl.instance a sym @[[topaSym]]
    inst a of a
    a.i.z.y.x <= ia.z.y.x
    a.i.a <= ia.a
    oa.z.y.x <= a.o.z.y.x
    oa.a <= a.o.a
    ; CHECK:      firrtl.instance b sym @[[topbSym]]
    inst b of b
    b.q.a.b.c <= ib.a.b.c
    b.q.z <= ib.z
    ob.a.b.c <= b.r.a.b.c
    ob.z <= b.r.z

  ; CHECK:      firrtl.module private @a(
  ; CHECK-SAME:   in %i: {{.+}} sym @[[aiSym]]
  ; CHECK-NOT:    out
  ; CHECK-SAME:     [#firrtl.subAnno<fieldID = 1, {circt.nonlocal = @[[nla_2]], class = "nla2"}>,
  ; CHECK-SAME:      #firrtl.subAnno<fieldID = 4, {circt.nonlocal = @[[nla_1]], class = "nla1"}>]
  module a:
    input i: {z: {y: {x: UInt<1>}}, a: UInt<1>}
    output o: {z: {y: {x: UInt<1>}}, a: UInt<1>}
    o <= i

  ; CHECK-NOT:  firrtl.module private @b(
  module b:
    input q: {a: {b: {c: UInt<1>}}, z: UInt<1>}
    output r: {a: {b: {c: UInt<1>}}, z: UInt<1>}
    r <= q
