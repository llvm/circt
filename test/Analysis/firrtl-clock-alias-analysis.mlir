// RUN: circt-opt %s -test-firrtl-gated-clock-conversion=print-clock-aliases -split-input-file -o /dev/null | FileCheck %s
// RUN: circt-opt %s -test-firrtl-gated-clock-conversion=verify-clock-aliases -split-input-file -o /dev/null

// A gate aliases its input; dead gates stay tracked.
// CHECK-LABEL: clock-aliases @Gates
// CHECK-NEXT:  clock-alias-class: base=Gates.clk members=[Gates.<firrtl.int.clock_gate#0>, Gates.<firrtl.int.clock_gate#0>, Gates.clk]
firrtl.circuit "Gates" {
  firrtl.module @Gates(in %clk: !firrtl.clock, in %en1: !firrtl.uint<1>,
                       in %en2: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %g1 = firrtl.int.clock_gate %clk, %en1
    %g2 = firrtl.int.clock_gate %g1, %en2
    %r1 = firrtl.reg %g2 : !firrtl.clock, !firrtl.uint<8>
    %r2 = firrtl.reg %g2 : !firrtl.clock, !firrtl.uint<8>
    %r3 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r1, %d : !firrtl.uint<8>
    firrtl.matchingconnect %r2, %d : !firrtl.uint<8>
    firrtl.matchingconnect %r3, %d : !firrtl.uint<8>
  }
}

// -----

// Wires and nodes are looked through. A cast of an integer is a base clock.
// CHECK-LABEL: clock-aliases @LookThrough
// CHECK-NEXT:  clock-alias-class: base=LookThrough.<firrtl.asClock#0> members=[LookThrough.<firrtl.asClock#0>]
// CHECK-NEXT:  clock-alias-class: base=LookThrough.<firrtl.asClock#0> members=[LookThrough.<firrtl.asClock#0>]
// CHECK-NEXT:  clock-alias-class: base=LookThrough.clk members=[LookThrough.clk, LookThrough.n, LookThrough.w]
firrtl.circuit "LookThrough" {
  firrtl.module @LookThrough(in %clk: !firrtl.clock, in %u: !firrtl.uint<1>,
                             in %d: !firrtl.uint<8>) {
    %w = firrtl.wire : !firrtl.clock
    firrtl.matchingconnect %w, %clk : !firrtl.clock
    %n = firrtl.node %w : !firrtl.clock
    %r1 = firrtl.reg %n : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r1, %d : !firrtl.uint<8>
    %c1 = firrtl.asClock %u : (!firrtl.uint<1>) -> !firrtl.clock
    %c2 = firrtl.asClock %u : (!firrtl.uint<1>) -> !firrtl.clock
    %r2 = firrtl.reg %c1 : !firrtl.clock, !firrtl.uint<8>
    %r3 = firrtl.reg %c2 : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r2, %d : !firrtl.uint<8>
    firrtl.matchingconnect %r3, %d : !firrtl.uint<8>
  }
}

// -----

// A single instance joins the callee's ports to the caller's class.
// CHECK-LABEL: clock-aliases @SingleInst
// CHECK-NEXT:  clock-alias-class: base=SingleInst.clk members=[Child.a, Child.b, SingleInst.c.a, SingleInst.c.b, SingleInst.clk]
firrtl.circuit "SingleInst" {
  firrtl.module @Child(in %a: !firrtl.clock, in %b: !firrtl.clock,
                       in %d: !firrtl.uint<8>) {
    %ra = firrtl.reg %a : !firrtl.clock, !firrtl.uint<8>
    %ra2 = firrtl.reg %a : !firrtl.clock, !firrtl.uint<8>
    %rb = firrtl.reg %b : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %ra, %d : !firrtl.uint<8>
    firrtl.matchingconnect %ra2, %d : !firrtl.uint<8>
    firrtl.matchingconnect %rb, %d : !firrtl.uint<8>
  }
  firrtl.module @SingleInst(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %a, %b, %cd = firrtl.instance c @Child(in a: !firrtl.clock, in b: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %a, %clk : !firrtl.clock
    firrtl.matchingconnect %b, %clk : !firrtl.clock
    firrtl.matchingconnect %cd, %d : !firrtl.uint<8>
  }
}

// -----

// Instances that agree on a port join its class.
// CHECK-LABEL: clock-aliases @Agree
// CHECK-NEXT:  clock-alias-class: base=Agree.clk members=[Agree.clk, Agree.i1.a, Agree.i1.b, Agree.i2.a, Agree.i2.b, Child.a, Child.b]
firrtl.circuit "Agree" {
  firrtl.module @Child(in %a: !firrtl.clock, in %b: !firrtl.clock,
                       in %d: !firrtl.uint<8>) {
    %ra = firrtl.reg %a : !firrtl.clock, !firrtl.uint<8>
    %ra2 = firrtl.reg %a : !firrtl.clock, !firrtl.uint<8>
    %rb = firrtl.reg %b : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %ra, %d : !firrtl.uint<8>
    firrtl.matchingconnect %ra2, %d : !firrtl.uint<8>
    firrtl.matchingconnect %rb, %d : !firrtl.uint<8>
  }
  firrtl.module @Agree(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %a1, %b1, %d1 = firrtl.instance i1 @Child(in a: !firrtl.clock, in b: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %a1, %clk : !firrtl.clock
    firrtl.matchingconnect %b1, %clk : !firrtl.clock
    firrtl.matchingconnect %d1, %d : !firrtl.uint<8>
    %a2, %b2, %d2 = firrtl.instance i2 @Child(in a: !firrtl.clock, in b: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %a2, %clk : !firrtl.clock
    firrtl.matchingconnect %b2, %clk : !firrtl.clock
    firrtl.matchingconnect %d2, %d : !firrtl.uint<8>
  }
}

// -----

// Instances that disagree on `b` keep it out of both classes.
// CHECK-LABEL: clock-aliases @Disagree
// CHECK-NEXT:  clock-alias-class: base=<none> members=[Child.b]
// CHECK-NEXT:  clock-alias-class: base=Disagree.c1 members=[Child.a, Disagree.c1, Disagree.i1.a, Disagree.i1.b, Disagree.i2.a]
// CHECK-NEXT:  clock-alias-class: base=Disagree.c2 members=[Disagree.c2, Disagree.i2.b]
firrtl.circuit "Disagree" {
  firrtl.module @Child(in %a: !firrtl.clock, in %b: !firrtl.clock,
                       in %d: !firrtl.uint<8>) {
    %ra = firrtl.reg %a : !firrtl.clock, !firrtl.uint<8>
    %ra2 = firrtl.reg %a : !firrtl.clock, !firrtl.uint<8>
    %rb = firrtl.reg %b : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %ra, %d : !firrtl.uint<8>
    firrtl.matchingconnect %ra2, %d : !firrtl.uint<8>
    firrtl.matchingconnect %rb, %d : !firrtl.uint<8>
  }
  firrtl.module @Disagree(in %c1: !firrtl.clock, in %c2: !firrtl.clock,
                          in %d: !firrtl.uint<8>) {
    %a1, %b1, %d1 = firrtl.instance i1 @Child(in a: !firrtl.clock, in b: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %a1, %c1 : !firrtl.clock
    firrtl.matchingconnect %b1, %c1 : !firrtl.clock
    firrtl.matchingconnect %d1, %d : !firrtl.uint<8>
    %a2, %b2, %d2 = firrtl.instance i2 @Child(in a: !firrtl.clock, in b: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %a2, %c1 : !firrtl.clock
    firrtl.matchingconnect %b2, %c2 : !firrtl.clock
    firrtl.matchingconnect %d2, %d : !firrtl.uint<8>
  }
}

// -----

// Each instance's output aliases its own input, but not the callee's ports.
// CHECK-LABEL: clock-aliases @PassThrough
// CHECK-NEXT:  clock-alias-class: base=<none> members=[Buf.i, Buf.o, Buf.w]
// CHECK-NEXT:  clock-alias-class: base=PassThrough.c1 members=[PassThrough.b1.i, PassThrough.b1.o, PassThrough.c1]
// CHECK-NEXT:  clock-alias-class: base=PassThrough.c2 members=[PassThrough.b2.i, PassThrough.b2.o, PassThrough.c2]
firrtl.circuit "PassThrough" {
  firrtl.module @Buf(in %i: !firrtl.clock, out %o: !firrtl.clock) {
    %w = firrtl.wire : !firrtl.clock
    firrtl.matchingconnect %w, %i : !firrtl.clock
    firrtl.matchingconnect %o, %w : !firrtl.clock
  }
  firrtl.module @PassThrough(in %c1: !firrtl.clock, in %c2: !firrtl.clock,
                             in %d: !firrtl.uint<8>) {
    %i1, %o1 = firrtl.instance b1 @Buf(in i: !firrtl.clock, out o: !firrtl.clock)
    firrtl.matchingconnect %i1, %c1 : !firrtl.clock
    %i2, %o2 = firrtl.instance b2 @Buf(in i: !firrtl.clock, out o: !firrtl.clock)
    firrtl.matchingconnect %i2, %c2 : !firrtl.clock
    %r1 = firrtl.reg %o1 : !firrtl.clock, !firrtl.uint<8>
    %r2 = firrtl.reg %o2 : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r1, %d : !firrtl.uint<8>
    firrtl.matchingconnect %r2, %d : !firrtl.uint<8>
  }
}

// -----

// Outputs generated inside a multiply instantiated module are distinct.
// CHECK-LABEL: clock-aliases @Osc
// CHECK-NEXT:  clock-alias-class: base=<none> members=[Osc.g1.o]
// CHECK-NEXT:  clock-alias-class: base=<none> members=[Osc.g2.o]
// CHECK-NEXT:  clock-alias-class: base=Gen.<firrtl.int.clock_div#0> members=[Gen.<firrtl.int.clock_div#0>, Gen.o]
// CHECK-NEXT:  clock-alias-class: base=Osc.clk members=[Gen.clk, Osc.clk, Osc.g1.clk, Osc.g2.clk]
firrtl.circuit "Osc" {
  firrtl.module @Gen(in %clk: !firrtl.clock, out %o: !firrtl.clock) {
    %c = firrtl.int.clock_div %clk by 1
    firrtl.matchingconnect %o, %c : !firrtl.clock
  }
  firrtl.module @Osc(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %c1, %o1 = firrtl.instance g1 @Gen(in clk: !firrtl.clock, out o: !firrtl.clock)
    firrtl.matchingconnect %c1, %clk : !firrtl.clock
    %c2, %o2 = firrtl.instance g2 @Gen(in clk: !firrtl.clock, out o: !firrtl.clock)
    firrtl.matchingconnect %c2, %clk : !firrtl.clock
    %r1 = firrtl.reg %o1 : !firrtl.clock, !firrtl.uint<8>
    %r2 = firrtl.reg %o2 : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r1, %d : !firrtl.uint<8>
    firrtl.matchingconnect %r2, %d : !firrtl.uint<8>
  }
}

// -----

// Muxes, dividers, inverters and top-level ports are base clocks.
// CHECK-LABEL: clock-aliases @NewBases
// CHECK-NEXT:  clock-alias-class: base=NewBases.<firrtl.int.clock_div#0> members=[NewBases.<firrtl.int.clock_div#0>]
// CHECK-NEXT:  clock-alias-class: base=NewBases.<firrtl.int.clock_inv#0> members=[NewBases.<firrtl.int.clock_inv#0>]
// CHECK-NEXT:  clock-alias-class: base=NewBases.<firrtl.mux#0> members=[NewBases.<firrtl.mux#0>]
// CHECK-NEXT:  clock-alias-class: base=NewBases.a members=[NewBases.a]
// CHECK-NEXT:  clock-alias-class: base=NewBases.b members=[NewBases.b]
firrtl.circuit "NewBases" {
  firrtl.module @NewBases(in %a: !firrtl.clock, in %b: !firrtl.clock,
                          in %s: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %m = firrtl.mux(%s, %a, %b) : (!firrtl.uint<1>, !firrtl.clock, !firrtl.clock) -> !firrtl.clock
    %div = firrtl.int.clock_div %a by 1
    %inv = firrtl.int.clock_inv %a
    %r1 = firrtl.reg %m : !firrtl.clock, !firrtl.uint<8>
    %r2 = firrtl.reg %div : !firrtl.clock, !firrtl.uint<8>
    %r3 = firrtl.reg %inv : !firrtl.clock, !firrtl.uint<8>
    %r4 = firrtl.reg %a : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r1, %d : !firrtl.uint<8>
    firrtl.matchingconnect %r2, %d : !firrtl.uint<8>
    firrtl.matchingconnect %r3, %d : !firrtl.uint<8>
    firrtl.matchingconnect %r4, %d : !firrtl.uint<8>
  }
}

// -----

// `D.q` joins `D.p` only once `C.o` is known to alias `C.i`.
// CHECK-LABEL: clock-aliases @SiblingDFirst
// CHECK-NEXT:  clock-alias-class: base=SiblingDFirst.clk members=[C.i, C.o, D.p, D.q, SiblingDFirst.c1.i, SiblingDFirst.c1.o, SiblingDFirst.c2.i, SiblingDFirst.c2.o, SiblingDFirst.clk, SiblingDFirst.dd.p, SiblingDFirst.dd.q]
firrtl.circuit "SiblingDFirst" {
  firrtl.module @D(in %p: !firrtl.clock, in %q: !firrtl.clock,
                   in %d: !firrtl.uint<8>) {
    %rp = firrtl.reg %p : !firrtl.clock, !firrtl.uint<8>
    %rq = firrtl.reg %q : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %rp, %d : !firrtl.uint<8>
    firrtl.matchingconnect %rq, %d : !firrtl.uint<8>
  }
  firrtl.module @C(in %i: !firrtl.clock, out %o: !firrtl.clock) {
    firrtl.matchingconnect %o, %i : !firrtl.clock
  }
  firrtl.module @SiblingDFirst(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %dp, %dq, %dd = firrtl.instance dd @D(in p: !firrtl.clock, in q: !firrtl.clock, in d: !firrtl.uint<8>)
    %ci1, %co1 = firrtl.instance c1 @C(in i: !firrtl.clock, out o: !firrtl.clock)
    %ci2, %co2 = firrtl.instance c2 @C(in i: !firrtl.clock, out o: !firrtl.clock)
    firrtl.matchingconnect %ci1, %clk : !firrtl.clock
    firrtl.matchingconnect %ci2, %clk : !firrtl.clock
    firrtl.matchingconnect %dp, %clk : !firrtl.clock
    firrtl.matchingconnect %dq, %co1 : !firrtl.clock
    firrtl.matchingconnect %dd, %d : !firrtl.uint<8>
  }
}

// -----

// A leaf port is unanimous only because its parent's port is.
// CHECK-LABEL: clock-aliases @ThreeLevel
// CHECK-NEXT:  clock-alias-class: base=ThreeLevel.clk members=[Leaf.clk, Mid.clk, Mid.l1.clk, Mid.l2.clk, ThreeLevel.clk, ThreeLevel.m1.clk, ThreeLevel.m2.clk]
firrtl.circuit "ThreeLevel" {
  firrtl.module @Leaf(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
  firrtl.module @Mid(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %c1, %d1 = firrtl.instance l1 @Leaf(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c1, %clk : !firrtl.clock
    firrtl.matchingconnect %d1, %d : !firrtl.uint<8>
    %c2, %d2 = firrtl.instance l2 @Leaf(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c2, %clk : !firrtl.clock
    firrtl.matchingconnect %d2, %d : !firrtl.uint<8>
  }
  firrtl.module @ThreeLevel(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %c1, %d1 = firrtl.instance m1 @Mid(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c1, %clk : !firrtl.clock
    firrtl.matchingconnect %d1, %d : !firrtl.uint<8>
    %c2, %d2 = firrtl.instance m2 @Mid(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c2, %clk : !firrtl.clock
    firrtl.matchingconnect %d2, %d : !firrtl.uint<8>
  }
}

// -----

// An appended input base port aliases the port it shadows even when the
// callers disagree.
// CHECK-LABEL: clock-aliases @BasePorts
// CHECK-NEXT:  clock-alias-class: base=BasePorts.clk members=[BasePorts.<firrtl.int.clock_gate#0>, BasePorts.<firrtl.int.clock_gate#0>, BasePorts.clk, BasePorts.i1._gatedClock_baseClock_clk, BasePorts.i1.clk, BasePorts.i2._gatedClock_baseClock_clk, BasePorts.i2.clk, BasePorts.i3._gatedClock_baseClock_clk, BasePorts.i3.clk, Child._gatedClock_baseClock_clk, Child.clk]
firrtl.circuit "BasePorts" {
  firrtl.module @Child(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
  firrtl.module @BasePorts(in %clk: !firrtl.clock, in %en1: !firrtl.uint<1>,
                           in %en2: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %g1 = firrtl.int.clock_gate %clk, %en1
    %g2 = firrtl.int.clock_gate %clk, %en2
    %c1, %d1 = firrtl.instance i1 @Child(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c1, %g1 : !firrtl.clock
    firrtl.matchingconnect %d1, %d : !firrtl.uint<8>
    %c2, %d2 = firrtl.instance i2 @Child(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c2, %clk : !firrtl.clock
    firrtl.matchingconnect %d2, %d : !firrtl.uint<8>
    %c3, %d3 = firrtl.instance i3 @Child(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c3, %g2 : !firrtl.clock
    firrtl.matchingconnect %d3, %d : !firrtl.uint<8>
  }
}

// -----

// An appended output base port aliases the port it shadows per instance.
// CHECK-LABEL: clock-aliases @OutBasePorts
// CHECK-NEXT:  clock-alias-class: base=<none> members=[Gen.<firrtl.int.clock_gate#0>, Gen._gatedClock_baseClock_gclk, Gen.clk, Gen.gclk]
// CHECK-NEXT:  clock-alias-class: base=OutBasePorts.a members=[OutBasePorts.a, OutBasePorts.x1._gatedClock_baseClock_gclk, OutBasePorts.x1.clk, OutBasePorts.x1.gclk]
// CHECK-NEXT:  clock-alias-class: base=OutBasePorts.b members=[OutBasePorts.b, OutBasePorts.x2._gatedClock_baseClock_gclk, OutBasePorts.x2.clk, OutBasePorts.x2.gclk]
firrtl.circuit "OutBasePorts" {
  firrtl.module @Gen(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                     out %gclk: !firrtl.clock) {
    %g = firrtl.int.clock_gate %clk, %en
    firrtl.matchingconnect %gclk, %g : !firrtl.clock
  }
  firrtl.module @OutBasePorts(in %a: !firrtl.clock, in %b: !firrtl.clock,
                              in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %c1, %e1, %g1 = firrtl.instance x1 @Gen(in clk: !firrtl.clock, in en: !firrtl.uint<1>, out gclk: !firrtl.clock)
    firrtl.matchingconnect %c1, %a : !firrtl.clock
    firrtl.matchingconnect %e1, %en : !firrtl.uint<1>
    %c2, %e2, %g2 = firrtl.instance x2 @Gen(in clk: !firrtl.clock, in en: !firrtl.uint<1>, out gclk: !firrtl.clock)
    firrtl.matchingconnect %c2, %b : !firrtl.clock
    firrtl.matchingconnect %e2, %en : !firrtl.uint<1>
    %r1 = firrtl.reg %g1 : !firrtl.clock, !firrtl.uint<8>
    %r2 = firrtl.reg %g2 : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r1, %d : !firrtl.uint<8>
    firrtl.matchingconnect %r2, %d : !firrtl.uint<8>
  }
}

// -----

// `r2`'s base clock is defined after it and reaches it through a new wire.
// CHECK-LABEL: clock-aliases @Wires
// CHECK-NEXT:  clock-alias-class: base=Wires.clk members=[Src.clk, Src.o, Wires.<firrtl.int.clock_gate#0>, Wires.<firrtl.int.clock_gate#0>, Wires.<firrtl.wire#0>, Wires.clk, Wires.s.clk, Wires.s.o, Wires.w1, Wires.w2]
firrtl.circuit "Wires" {
  firrtl.module @Src(in %clk: !firrtl.clock, out %o: !firrtl.clock) {
    firrtl.matchingconnect %o, %clk : !firrtl.clock
  }
  firrtl.module @Wires(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                          in %d: !firrtl.uint<8>) {
    %g = firrtl.int.clock_gate %clk, %en
    %w1 = firrtl.wire : !firrtl.clock
    firrtl.matchingconnect %w1, %g : !firrtl.clock
    %r1 = firrtl.reg %w1 : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r1, %d : !firrtl.uint<8>
    %w2 = firrtl.wire : !firrtl.clock
    %r2 = firrtl.reg %w2 : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r2, %d : !firrtl.uint<8>
    %si, %so = firrtl.instance s @Src(in clk: !firrtl.clock, out o: !firrtl.clock)
    firrtl.matchingconnect %si, %clk : !firrtl.clock
    %g2 = firrtl.int.clock_gate %so, %en
    firrtl.matchingconnect %w2, %g2 : !firrtl.clock
  }
}
