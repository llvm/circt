; RUN: firtool -split-input-file -ir-fir --preserve-aggregate=all -disable-opt %s | FileCheck %s
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
; CHECK-LABEL: firrtl.circuit "Top0"
FIRRTL version 4.0.0
circuit Top0 :
  ; CHECK: firrtl.module @Top0
  public module Top0 :
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
; CHECK-LABEL: firrtl.circuit "Top1"
FIRRTL version 4.0.0
circuit Top1 :
  ; CHECK: firrtl.module @Top1
  public module Top1 :
    ; CHECK-COUNT-2: firrtl.instance {{a(1|2)}} @A(
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
; CHECK-LABEL: firrtl.circuit "Top2"
FIRRTL version 4.0.0
circuit Top2 :
  ; CHECK: firrtl.module @Top2
  public module Top2 :
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
; CHECK-LABEL: firrtl.circuit "Top3"
FIRRTL version 4.0.0
circuit Top3 :
  ; CHECK: firrtl.module @Top3
  public module Top3 :
    ; CHECK-COUNT-2: firrtl.instance {{a(1|2)}} @A(
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
; CHECK-LABEL: firrtl.circuit "Top4"
FIRRTL version 4.0.0
circuit Top4 :
  ; CHECK: firrtl.module @Top4
  public module Top4 :
    output out: UInt<1>
    ; CHECK-COUNT-2: firrtl.instance {{a(1|2)}} @A(
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
; CHECK-LABEL: firrtl.circuit "Top5"
FIRRTL version 4.0.0
circuit Top5 :
  ; CHECK: firrtl.module @Top5
  public module Top5 :
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
; CHECK-LABEL: firrtl.circuit "Top6"
FIRRTL version 4.0.0
circuit Top6 :
  ; CHECK: firrtl.module @Top6
  public module Top6 :
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
; "Modules with aggregate ports that are connected should NOT dedup if their
; port names differ".
;
; Since the MFC support expanding connects and partial connects when the port
; names differ, this now testing that the modules succesfully dedup.
;
; CHECK-LABEL: firrtl.circuit "FooAndBarModule0"
FIRRTL version 4.0.0
circuit FooAndBarModule0 :
  ; CHECK: firrtl.module private @FooModule
  module FooModule :
    output io : {flip foo : UInt<1>, fuzz : UInt<1>}
    io.fuzz <= io.foo
  ; CHECK-NOT: firrtl.module private @BarModule
  module BarModule :
    output io : {flip bar : UInt<1>, buzz : UInt<1>}
    io.buzz <= io.bar
  ; CHECK: firrtl.module @FooAndBarModule0
  public module FooAndBarModule0 :
    output io : {foo : {flip foo : UInt<1>, fuzz : UInt<1>}, bar : {flip bar : UInt<1>, buzz : UInt<1>}}
    ; CHECK: firrtl.instance foo @FooModule
    ; CHECK: firrtl.instance bar @FooModule
    inst foo of FooModule
    inst bar of BarModule
    io.foo <= foo.io
    io.bar <= bar.io

; // -----
; "Modules with aggregate ports that are connected should dedup if their port
; names are the same".
;
; CHECK-LABEL: firrtl.circuit "FooAndBarModule1"
FIRRTL version 4.0.0
circuit FooAndBarModule1 :
  ; CHECK: firrtl.module private @FooModule
  module FooModule :
    output io : {flip foo : UInt<1>, fuzz : UInt<1>}
    io.fuzz <= io.foo
  ; CHECK-NOT: firrtl.module private @BarModule
  module BarModule :
    output io : {flip foo : UInt<1>, fuzz : UInt<1>}
    io.fuzz <= io.foo
  ; CHECK: firrtl.module @FooAndBarModule1
  public module FooAndBarModule1 :
    output io : {foo : {flip foo : UInt<1>, fuzz : UInt<1>}, bar : {flip foo : UInt<1>, fuzz : UInt<1>}}
    ; CHECK-COUNT-2: firrtl.instance {{(foo|bar)}} @FooModule
    inst foo of FooModule
    inst bar of BarModule
    io.foo <= foo.io
    io.bar <= bar.io

; // -----
; "The module A and B should be deduped with the first module in order".
;
; CHECK-LABEL: firrtl.circuit "Top7"
FIRRTL version 4.0.0
circuit Top7 :
  ; CHECK: firrtl.module @Top7
  public module Top7 :
    ; CHECK-COUNT-2: firrtl.instance {{a(1|2)}} @A(
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
; CHECK-LABEL: firrtl.circuit "Top8"
FIRRTL version 4.0.0
circuit Top8 :
  ; CHECK: firrtl.module @Top8
  public module Top8 :
    ; CHECK-COUNT-2: firrtl.instance {{a(1|2)}} @A(
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
; CHECK-LABEL: firrtl.circuit "Top9"
FIRRTL version 4.0.0
circuit Top9 : %[[
  {
    "class":"circt.test",
    "target":"Top9.A.b"
  }
]]
  ; CHECK: hw.hierpath private @[[nlaSym:[_a-zA-Z0-9]+]] [@Top9::@[[a1Sym:[_a-zA-Z0-9]+]], @A]
  ; CHECK: firrtl.module @Top9
  ; CHECK-NEXT: firrtl.instance a1 sym @[[a1Sym]] @A(
  ; CHECK-NEXT: firrtl.instance a2 @A(
  public module Top9 :
    inst a1 of A
    inst a2 of A_
  ; CHECK: module private @A(
  module A :
    output x: UInt<1>
    ; CHECK: firrtl.node %c0_ui1 {annotations = [{circt.nonlocal = @[[nlaSym]], class = "circt.test"}]}
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
; CHECK-LABEL: firrtl.circuit "Top10"
FIRRTL version 4.0.0
circuit Top10 : %[[
  {
    "class":"circt.test",
    "data":"hello",
    "target":"Top10.A.b"
  },
  {
    "class":"circt.test",
    "data":"world",
    "target":"~Top10|A_>b"
  }
]]
  ; CHECK: hw.hierpath private @[[nlaSym2:[_a-zA-Z0-9]+]] [@Top10::@[[a2Sym:[_a-zA-Z0-9]+]], @A]
  ; CHECK: hw.hierpath private @[[nlaSym1:[_a-zA-Z0-9]+]] [@Top10::@[[a1Sym:[_a-zA-Z0-9]+]], @A]
  ; CHECK: firrtl.module @Top10
  ; CHECK-NEXT: firrtl.instance a1 sym @[[a1Sym]] @A(
  ; CHECK-NEXT: firrtl.instance a2 sym @[[a2Sym]] @A(
  public module Top10 :
    inst a1 of A
    inst a2 of A_
  ; CHECK: module private @A(
  module A :
    output x: UInt<1>
    ; CHECK: firrtl.node %c0_ui1 {annotations = [{circt.nonlocal = @[[nlaSym1]], class = "circt.test", data = "hello"}, {circt.nonlocal = @[[nlaSym2]], class = "circt.test", data = "world"}]}
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
; different names".
;
; CHECK-LABEL: firrtl.circuit "Top11"
FIRRTL version 4.0.0
circuit Top11 : %[[
  {
    "class":"circt.test",
    "data":"hello",
    "target":"~Top11|A>a"
  },
  {
    "class":"circt.test",
    "data":"hello",
    "target":"~Top11|A_>b"
  }
]]
  ; CHECK-NOT: hw.hierpath
  ; CHECK: hw.hierpath private @[[nlaSym2:[_a-zA-Z0-9]+]] [@Top11::@[[a1sym:[_a-zA-Z0-9]+]], @A]
  ; CHECK: hw.hierpath private @[[nlaSym1:[_a-zA-Z0-9]+]] [@Top11::@[[a1Sym]], @A]
  public module Top11 :
    inst a1 of A
    a1 is invalid
    inst a2 of A_
    a2 is invalid
  ; CHECK: firrtl.module private @A(
  module A :
    input x: UInt<1>
    output y: UInt<1>
    ; CHECK: firrtl.node {{.+}} {annotations = [{circt.nonlocal = @[[nlaSym1]], class = "circt.test", data = "hello"}, {circt.nonlocal = @[[nlaSym2]], class = "circt.test", data = "hello"}]}
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
FIRRTL version 4.0.0
circuit main:
  module dupe:
    input in: UInt<8>
    output out: UInt<8>
    out <= in
  ; CHECK: firrtl.module @main
  public module main:
    input in:  UInt<8>
    output out: UInt<8>
    out <= in

; // -----
; "The deduping module A and A_ should rename instances and signals that have different names"
;
; CHECK-LABEL: firrtl.circuit "Top12"
FIRRTL version 4.0.0
circuit Top12 : %[[
  {
    "class":"circt.test",
    "data":"B",
    "target":"~Top12|Top12/a:A/b:B"
  },
  {
    "class":"circt.test",
    "data":"B.foo",
    "target":"~Top12|Top12/a:A/b:B>foo"
  },
  {
    "class":"circt.test",
    "data":"B_",
    "target":"~Top12|Top12/a_:A_/b_:B_"
  },
  {
    "class":"circt.test",
    "data":"B_.bar",
    "target":"~Top12|Top12/a_:A_/b_:B_>bar"
  }
]]
  ; CHECK: hw.hierpath private @[[nla_a:[_a-zA-Z0-9]+]] [@Top12::@[[aSym:[_a-zA-Z0-9]+]], @A::@[[bSym:[_a-zA-Z0-9]+]], @B]
  ; CHECK: hw.hierpath private @[[nla_a_:[_a-zA-Z0-9]+]] [@Top12::@[[a_Sym:[_a-zA-Z0-9]+]], @A::@[[bSym]], @B]
  ; CHECK: firrtl.module @Top12
  public module Top12 :
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
  ; CHECK-SAME: {circt.nonlocal = @[[nla_a]], class = "circt.test", data = "B"}
  ; CHECK-SAME: {circt.nonlocal = @[[nla_a_]], class = "circt.test", data = "B_"}
  module B :
    ; CHECK: firrtl.node
    ; CHECK-SAME: {circt.nonlocal = @[[nla_a]], class = "circt.test", data = "B.foo"}
    ; CHECK-SAME: {circt.nonlocal = @[[nla_a_]], class = "circt.test", data = "B_.bar"}
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
; CHECK-LABEL: firrtl.circuit "Top13"
FIRRTL version 4.0.0
circuit Top13 : %[[
  {
    "class":"circt.test",
    "data":"one",
    "target":"~Top13|Top13/a:A/b:B/c:C/d:D"
  },
  {
    "class":"circt.test",
    "data":"one",
    "target":"~Top13|Top13/a:A/b:B/c:C/d:D>foo"
  },
  {
    "class":"circt.test",
    "data":"two",
    "target":"~Top13|Top13/a_:A_/b_:B_/c_:C_/d_:D_"
  },
  {
    "class":"circt.test",
    "data":"two",
    "target":"~Top13|Top13/a_:A_/b_:B_/c_:C_/d_:D_>bar"
  }
]]
  ; CHECK:        hw.hierpath private @[[nla_a:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top13::@[[aSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @A::@[[bSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @B::@[[cSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @C::@[[dSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @D]
  ; CHECK-NEXT:   hw.hierpath private @[[nla_a_:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top13::@[[a_Sym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @A::@[[bSym]],
  ; CHECK-SAME:      @B::@[[cSym]],
  ; CHECK-SAME:      @C::@[[dSym]],
  ; CHECK-SAME:      @D]
  ; CHECK-NEXT:   firrtl.module @Top13
  public module Top13 :
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
; CHECK-LABEL: firrtl.circuit "Top14"
FIRRTL version 4.0.0
circuit Top14 : %[[
  {
    "class":"circt.test",
    "data":"nla1",
    "target":"~Top14|Top14/b:B"
  },
  {
    "class":"circt.test",
    "data":"nla2",
    "target":"~Top14|Top14/b_:B_"
  },
  {
    "class":"circt.test",
    "data":"nla3",
    "target":"~Top14|Top14/a1:A/b_:B_"
  },
  {
    "class":"circt.test",
    "data":"nla4",
    "target":"~Top14|Top14/a2:A/b_:B_"
  },
  {
    "class":"circt.test",
    "data":"nla5",
    "target":"~Top14|Top14/a1:A/b:B"
  },
  {
    "class":"circt.test",
    "data":"nla6",
    "target":"~Top14|Top14/a2:A/b:B"
  },
  {
    "class":"circt.test",
    "data":"nla7",
    "target":"~Top14|Top14/b:B/c:C"
  },
  {
    "class":"circt.test",
    "data":"nla8",
    "target":"~Top14|Top14/b_:B_/c:C"
  },
  {
    "class":"circt.test",
    "data":"nla9",
    "target":"~Top14|Top14/a1:A/b_:B_/c:C"
  },
  {
    "class":"circt.test",
    "data":"nla10",
    "target":"~Top14|Top14/a2:A/b_:B_/c:C"
  },
  {
    "class":"circt.test",
    "data":"nla11",
    "target":"~Top14|Top14/a1:A/b:B/c:C"
  },
  {
    "class":"circt.test",
    "data":"nla12",
    "target":"~Top14|Top14/a2:A/b:B/c:C"
  }
]]
  ; CHECK:        hw.hierpath private @[[nla_1:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[TopbSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @B]
  ; CHECK:        hw.hierpath private @[[nla_2:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[Topb_Sym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @B]
  ; CHECK:        hw.hierpath private @[[nla_3:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[Topa1Sym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @A::@[[Ab_Sym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @B]
  ; CHECK:        hw.hierpath private @[[nla_4:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[Topa2Sym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @A::@[[Ab_Sym]],
  ; CHECK-SAME:      @B]
  ; CHECK:        hw.hierpath private @[[nla_5:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[Topa1Sym]],
  ; CHECK-SAME:      @A::@[[AbSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @B]
  ; CHECK:        hw.hierpath private @[[nla_6:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[Topa2Sym]],
  ; CHECK-SAME:      @A::@[[AbSym]],
  ; CHECK-SAME:      @B]
  ; CHECK:        hw.hierpath private @[[nla_7:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[TopbSym]],
  ; CHECK-SAME:      @B::@[[BcSym:[_a-zA-Z0-9]+]],
  ; CHECK-SAME:      @C]
  ; CHECK:        hw.hierpath private @[[nla_8:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[Topb_Sym]],
  ; CHECK-SAME:      @B::@[[BcSym]],
  ; CHECK-SAME:      @C]
  ; CHECK:        hw.hierpath private @[[nla_9:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[Topa1Sym]],
  ; CHECK-SAME:      @A::@[[Ab_Sym]],
  ; CHECK-SAME:      @B::@[[BcSym]],
  ; CHECK-SAME:      @C]
  ; CHECK:        hw.hierpath private @[[nla_10:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[Topa2Sym]],
  ; CHECK-SAME:      @A::@[[Ab_Sym]],
  ; CHECK-SAME:      @B::@[[BcSym]],
  ; CHECK-SAME:      @C]
  ; CHECK:        hw.hierpath private @[[nla_11:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[Topa1Sym]],
  ; CHECK-SAME:      @A::@[[AbSym]],
  ; CHECK-SAME:      @B::@[[BcSym]],
  ; CHECK-SAME:      @C]
  ; CHECK:        hw.hierpath private @[[nla_12:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:     [@Top14::@[[Topa2Sym]],
  ; CHECK-SAME:      @A::@[[AbSym]],
  ; CHECK-SAME:      @B::@[[BcSym]],
  ; CHECK-SAME:      @C]
  ; CHECK-NEXT:   firrtl.module @Top14
  public module Top14 :
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
  ; CHECK-SAME:     [{circt.nonlocal = @[[nla_1]], class = "circt.test", data = "nla1"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_5]], class = "circt.test", data = "nla5"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_6]], class = "circt.test", data = "nla6"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_2]], class = "circt.test", data = "nla2"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_3]], class = "circt.test", data = "nla3"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_4]], class = "circt.test", data = "nla4"}]
  module B :
    ; CHECK-NEXT:   firrtl.instance c sym @[[BcSym]]
    ; CHECK-SAME:     @C()
    inst c of C

  ; CHECK:        firrtl.module private @C()
  ; CHECK-SAME:     [{circt.nonlocal = @[[nla_7]], class = "circt.test", data = "nla7"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_8]], class = "circt.test", data = "nla8"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_9]], class = "circt.test", data = "nla9"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_10]], class = "circt.test", data = "nla10"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_11]], class = "circt.test", data = "nla11"},
  ; CHECK-SAME:      {circt.nonlocal = @[[nla_12]], class = "circt.test", data = "nla12"}]
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
; CHECK-LABEL: firrtl.circuit "Top15"
FIRRTL version 4.0.0
circuit Top15 : %[[
  {
    "class":"circt.test",
    "data":"nla1",
    "target":"~Top15|a>i.a"
  },
  {
    "class":"circt.test",
    "data":"nla2",
    "target":"~Top15|b>q.a"
  }
]]
  ; CHECK:      hw.hierpath private @[[nla_2:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:   [@Top15::@[[topbSym:[_a-zA-Z0-9]+]], @a]
  ; CHECK:      hw.hierpath private @[[nla_1:[_a-zA-Z0-9]+]]
  ; CHECK-SAME:   [@Top15::@[[topaSym:[_a-zA-Z0-9]+]], @a]
  ; CHECK: firrtl.module @Top15
  public module Top15:
    input ia: {z: {y: {x: UInt<1>}}, a: UInt<1> un: UInt<2>}
    input ib: {a: {b: {c: UInt<1>}}, z: UInt<1> un: UInt<2>}
    output oa: {z: {y: {x: UInt<1>}}, a: UInt<1> un: UInt<2>}
    output ob: {a: {b: {c: UInt<1>}}, z: UInt<1> un: UInt<2>}
    ; CHECK:      firrtl.instance a sym @[[topaSym]]
    inst a of a
    a.i.z.y.x <= ia.z.y.x
    a.i.a <= ia.a
    a.i.un <= ia.un
    oa.z.y.x <= a.o.z.y.x
    oa.a <= a.o.a
    oa.un <= a.o.un

    ; CHECK:      firrtl.instance b sym @[[topbSym]]
    inst b of b
    b.q.un.b.c <= ib.a.b.c
    b.q.z <= ib.z
    b.q.a <= ib.un
    ob.a.b.c <= b.r.un.b.c
    ob.z <= b.r.z
    ob.un <= b.r.a

  ; CHECK:      firrtl.module private @a(
  ; CHECK-SAME:   in %i:
  ; CHECK-NOT:    out
  ; CHECK-SAME:     [{circt.fieldID = 4 : i32, circt.nonlocal = @[[nla_1]], class = "circt.test", data = "nla1"},
  ; CHECK-SAME:      {circt.fieldID = 5 : i32, circt.nonlocal = @[[nla_2]], class = "circt.test", data = "nla2"}]
  module a:
    input i: {z: {y: {x: UInt<1>}}, a: UInt<1>, un: UInt<2>}
    output o: {z: {y: {x: UInt<1>}}, a: UInt<1>, un: UInt<2>}
    o <= i

  ; CHECK-NOT:  firrtl.module private @b(
  module b:
    input  q: {un: {b: {c: UInt<1>}}, z: UInt<1>, a: UInt<2>}
    output r: {un: {b: {c: UInt<1>}}, z: UInt<1>, a: UInt<2>}
    r <= q

; // -----
; "The module A and A_ should be deduped even with different port names and info,
; and annotations should be remapped".
;
; CHECK-LABEL: firrtl.circuit "Top16"
FIRRTL version 4.0.0
circuit Top16 : %[[
  {
    "class": "circt.test",
     "target": "Top16.Top16.a2.y",
     "pin": "pin"
  }
]]
  ; CHECK: firrtl.module @Top16
  public module Top16 :
    output out: UInt<1>
    ; CHECK: instance a1 @A(
    ; CHECK-NOT: SourceAnnotation
    ; CHECK: instance a2 {{.+}} @A(
    inst a1 of A
    inst a2 of A_
    out <= and(a1.x, a2.y)
  ; CHECK: module private @A
  ; CHECK-SAME: [{circt.nonlocal = {{[^,]+}}, class = "circt.test", pin = "pin"}]
  module A : @[yy 2:2]
    output x: UInt<1> @[yy 2:2]
    x <= UInt(1)
  module A_ : @[xx 1:1]
    output y: UInt<1> @[xx 1:1]
    y <= UInt(1)

; CHECK-NOT: firrtl.module private @A_


; // -----
; "The module A and A_ should dedup with different annotation targets."
;
; This test is checking that a "DontTouchAnnotation" has the same behavior as
; any other annotation due to deduplication.
;
; CHECK-LABEL: firrtl.circuit "Top17"
FIRRTL version 4.0.0
circuit Top17 : %[[
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Top17|Top17/a1:A>b"
  }
]]
  ; CHECK: firrtl.module @Top17
  ; CHECK-NEXT: firrtl.instance a1 @A(
  ; CHECK-NEXT: firrtl.instance a2 @A(
  public module Top17 :
    inst a1 of A
    inst a2 of A_
  ; CHECK: module private @A(
  module A :
    output x: UInt<1>
    ; CHECK: firrtl.node %c0_ui1 {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
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
; "The module A and A_ should dedup with the same annotation targets."
;
; This is the same situation as the previous test, except that two
; "DontTouchAnnotation"s apply to the same deduplicated wire.  It is arguable
; that generating a symbol here is fine.  However, this is different from the
; Scala FIRRTL Compiler and we should make a decision on what to do with this.
;
; CHECK-LABEL: firrtl.circuit "Top18"
FIRRTL version 4.0.0
circuit Top18 : %[[
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"Top18.A.b"
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Top18|A_>b"
  }
]]
  ; CHECK: firrtl.module @Top18
  ; CHECK-NEXT: firrtl.instance a1 @A(
  ; CHECK-NEXT: firrtl.instance a2 @A(
  public module Top18 :
    inst a1 of A
    inst a2 of A_
  ; CHECK: module private @A(
  module A :
    output x: UInt<1>
    ; CHECK: firrtl.node
    ; CHECK-SAME: [{class = "firrtl.transforms.DontTouchAnnotation"}]
    wire b: UInt<1>
    b is invalid
    x <= b
  module A_ :
    output x: UInt<1>
    wire b: UInt<1>
    b is invalid
    x <= b

; CHECK-NOT: firrtl.module private @A_
