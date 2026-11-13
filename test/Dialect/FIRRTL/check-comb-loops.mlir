// RUN: circt-opt -allow-unregistered-dialect --pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-comb-loops))' --split-input-file --verify-diagnostics %s | FileCheck %s

// Loop-free circuit
// CHECK: firrtl.circuit "hasnoloops"
firrtl.circuit "hasnoloops"   {
  firrtl.module @thru(in %in1: !firrtl.uint<1>, in %in2: !firrtl.uint<1>, out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %out1, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %a, %in1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out2, %in2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @hasnoloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %x = firrtl.wire  : !firrtl.uint<1>
    %inner_in1, %inner_in2, %inner_out1, %inner_out2 = firrtl.instance inner @thru(in in1: !firrtl.uint<1>, in in2: !firrtl.uint<1>, out out1: !firrtl.uint<1>, out out2: !firrtl.uint<1>)
    firrtl.connect %inner_in1, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %inner_out1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %inner_in2, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %inner_out2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Simple combinational loop
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{y <- z <- y}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Single-element combinational loop
// CHECK-NOT: firrtl.circuit "loop"
firrtl.circuit "loop"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: loop.{w <- w}}}
  firrtl.module @loop(out %y: !firrtl.uint<8>) {
    %w = firrtl.wire  : !firrtl.uint<8>
    firrtl.connect %w, %w : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Node combinational loop
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{y <- z <- ... <- y}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %t = firrtl.and %c, %y : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %z = firrtl.node %t  : !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Combinational loop through a combinational memory read port
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{m.r.addr <- y <- z <- m.r.data <- m.r.addr}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %m_r = firrtl.mem Undefined  {depth = 2 : i64, name = "m", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %0 = firrtl.subfield %m_r[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %0, %clk : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %m_r[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %1, %y : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %m_r[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    firrtl.connect %2, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %3 = firrtl.subfield %m_r[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %z, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

// Combination loop through an instance
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  firrtl.module @thru(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{inner.in <- y <- z <- inner.out <- inner.in}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %inner_in, %inner_out = firrtl.instance inner @thru(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inner_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Multiple simple loops in one SCC
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{b <- ... <- d <- ... <- e <- b}}}
  firrtl.module @hasloops(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %a = firrtl.wire  : !firrtl.uint<1>
    %b = firrtl.wire  : !firrtl.uint<1>
    %c = firrtl.wire  : !firrtl.uint<1>
    %d = firrtl.wire  : !firrtl.uint<1>
    %e = firrtl.wire  : !firrtl.uint<1>
    %0 = firrtl.and %c, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.and %a, %d : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.and %c, %e : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %d, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %e, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %o, %e : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

firrtl.circuit "matchingConnectAndConnect" {
  // expected-error @below {{matchingConnectAndConnect.{a <- b <- a}}}
  firrtl.module @matchingConnectAndConnect(out %a: !firrtl.uint<11>, out %b: !firrtl.uint<11>) {
    %w = firrtl.wire : !firrtl.uint<11>
    firrtl.matchingconnect %b, %w : !firrtl.uint<11>
    firrtl.connect %a, %b : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.matchingconnect %b, %a : !firrtl.uint<11>
  }
}

// -----

firrtl.circuit "outputPortCycle"   {
  // expected-error @below {{outputPortCycle.{reg[0].a <- w.a <- reg[0].a}}}
  firrtl.module @outputPortCycle(out %reg: !firrtl.vector<bundle<a: uint<8>>, 2>) {
    %0 = firrtl.subindex %reg[0] : !firrtl.vector<bundle<a: uint<8>>, 2>
    %1 = firrtl.subindex %reg[0] : !firrtl.vector<bundle<a: uint<8>>, 2>
    %w = firrtl.wire : !firrtl.bundle<a:uint<8>>
    firrtl.connect %w, %0 : !firrtl.bundle<a:uint<8>>, !firrtl.bundle<a:uint<8>>
    firrtl.connect %1, %w : !firrtl.bundle<a:uint<8>>, !firrtl.bundle<a:uint<8>>
  }
}

// -----

firrtl.circuit "outputRead"   {
  firrtl.module @outputRead(out %reg: !firrtl.vector<bundle<a: uint<8>>, 2>) {
    %0 = firrtl.subindex %reg[0] : !firrtl.vector<bundle<a: uint<8>>, 2>
    %1 = firrtl.subindex %reg[1] : !firrtl.vector<bundle<a: uint<8>>, 2>
    firrtl.connect %1, %0 : !firrtl.bundle<a:uint<8>>, !firrtl.bundle<a:uint<8>>
  }
}

// -----

firrtl.circuit "vectorRegInit"   {
  firrtl.module @vectorRegInit(in %clk: !firrtl.clock) {
    %reg = firrtl.reg %clk : !firrtl.clock, !firrtl.vector<bundle<a: uint<8>>, 2>
    %0 = firrtl.subindex %reg[0] : !firrtl.vector<bundle<a: uint<8>>, 2>
    firrtl.connect %0, %0 : !firrtl.bundle<a:uint<8>>, !firrtl.bundle<a:uint<8>>
  }
}

// -----

firrtl.circuit "bundleRegInit"   {
  firrtl.module @bundleRegInit(in %clk: !firrtl.clock) {
    %reg = firrtl.reg %clk : !firrtl.clock, !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %reg[a] : !firrtl.bundle<a: uint<1>>
    firrtl.connect %0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "PortReadWrite"  {
  firrtl.extmodule private @Bar(in a: !firrtl.uint<1>)
  // expected-error @below {{PortReadWrite.{a <- bar.a <- a}}}
  firrtl.module @PortReadWrite() {
    %a = firrtl.wire : !firrtl.uint<1>
    %bar_a = firrtl.instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    firrtl.matchingconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %a, %bar_a : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo"  {
  firrtl.module private @Bar(in %a: !firrtl.uint<1>) {}
  // expected-error @below {{Foo.{a <- bar.a <- a}}}
  firrtl.module @Foo(out %a: !firrtl.uint<1>) {
    %bar_a = firrtl.instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    firrtl.matchingconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %a, %bar_a : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "outputPortCycle"   {
  firrtl.module private @Bar(in %a: !firrtl.bundle<a: uint<8>, b: uint<4>>) {}
  // expected-error @below {{outputPortCycle.{bar.a.a <- port[0].a <- bar.a.a}}}
  firrtl.module @outputPortCycle(out %port: !firrtl.vector<bundle<a: uint<8>, b: uint<4>>, 2>) {
    %0 = firrtl.subindex %port[0] : !firrtl.vector<bundle<a: uint<8>, b: uint<4>>, 2>
    %1 = firrtl.subindex %port[0] : !firrtl.vector<bundle<a: uint<8>, b: uint<4>>, 2>
    %w = firrtl.instance bar interesting_name  @Bar(in a: !firrtl.bundle<a: uint<8>, b: uint<4>>)
    firrtl.connect %w, %0 : !firrtl.bundle<a: uint<8>, b: uint<4>>, !firrtl.bundle<a: uint<8>, b: uint<4>>
    firrtl.connect %1, %w : !firrtl.bundle<a: uint<8>, b: uint<4>>, !firrtl.bundle<a: uint<8>, b: uint<4>>
  }
}

// -----

// Node combinational loop through vector subindex
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{w[3] <- z <- ... <- w[3]}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %w = firrtl.wire  : !firrtl.vector<uint<1>,10>
    %y = firrtl.subindex %w[3]  : !firrtl.vector<uint<1>,10>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %0 = firrtl.and %c, %y : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %z = firrtl.node %0  : !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Node combinational loop through vector subindex
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{b[0] <- bar_b[0] <- bar_a[0] <- b[0]}}}
  firrtl.module @hasloops(out %b: !firrtl.vector<uint<1>, 2>) {
    %bar_a = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %bar_b = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %1, %0 : !firrtl.uint<1>
    %4 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %5, %4 : !firrtl.uint<1>
    %v0 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    %v1 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %v1, %v0 : !firrtl.uint<1>
  }
}

// -----

// Combinational loop through instance ports
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasLoops"  {
  // expected-error @below {{hasLoops.{b[0] <- bar.b[0] <- bar.a[0] <- b[0]}}}
  firrtl.module @hasLoops(out %b: !firrtl.vector<uint<1>, 2>) {
    %bar_a, %bar_b = firrtl.instance bar  @Bar(in a: !firrtl.vector<uint<1>, 2>, out b: !firrtl.vector<uint<1>, 2>)
    %0 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %1, %0 : !firrtl.uint<1>
    %4 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %5, %4 : !firrtl.uint<1>
  }

  firrtl.module private @Bar(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %a[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %1, %0 : !firrtl.uint<1>
    %2 = firrtl.subindex %a[1] : !firrtl.vector<uint<1>, 2>
    %3 = firrtl.subindex %b[1] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %3, %2 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "bundleWire"   {
  // expected-error @below {{bundleWire.{out2 <- x <- w.foo.bar.baz <- out2}}}
  firrtl.module @bundleWire(in %arg: !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>,
                           out %out1: !firrtl.uint<1>, out %out2: !firrtl.sint<64>) {

    %w = firrtl.wire : !firrtl.bundle<foo: bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>>
    %w0 = firrtl.subfield %w[foo] : !firrtl.bundle<foo: bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>>
    %w0_0 = firrtl.subfield %w0[bar] : !firrtl.bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>
    %w0_0_0 = firrtl.subfield %w0_0[baz] : !firrtl.bundle<baz: sint<64>>
    %x = firrtl.wire  : !firrtl.sint<64>

    %0 = firrtl.subfield %arg[foo] : !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>
    %1 = firrtl.subfield %0[bar] : !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    %2 = firrtl.subfield %1[baz] : !firrtl.bundle<baz: uint<1>>
    %3 = firrtl.subfield %0[qux] : !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    firrtl.connect %w0_0_0, %3 : !firrtl.sint<64>, !firrtl.sint<64>
    firrtl.connect %x, %w0_0_0 : !firrtl.sint<64>, !firrtl.sint<64>
    firrtl.connect %out2, %x : !firrtl.sint<64>, !firrtl.sint<64>
    firrtl.connect %w0_0_0, %out2 : !firrtl.sint<64>, !firrtl.sint<64>
    firrtl.connect %out1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Combinational loop through instance ports
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasLoops"  {
  // expected-error @below {{hasLoops.{b[0] <- bar.b[0] <- bar.a[0] <- b[0]}}}
  firrtl.module @hasLoops(out %b: !firrtl.vector<uint<1>, 2>) {
    %bar_a, %bar_b = firrtl.instance bar  @Bar(in a: !firrtl.vector<uint<1>, 2>, out b: !firrtl.vector<uint<1>, 2>)
    %0 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %1, %0 : !firrtl.uint<1>
    %4 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %5, %4 : !firrtl.uint<1>
  }

  firrtl.module private @Bar(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    firrtl.matchingconnect %b, %a : !firrtl.vector<uint<1>, 2>
  }
}

// -----

firrtl.circuit "registerLoop"   {
  // CHECK: firrtl.module @registerLoop(in %clk: !firrtl.clock)
  firrtl.module @registerLoop(in %clk: !firrtl.clock) {
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %w[a]: !firrtl.bundle<a: uint<1>>
    %1 = firrtl.subfield %w[a]: !firrtl.bundle<a: uint<1>>
    %2 = firrtl.subfield %r[a]: !firrtl.bundle<a: uint<1>>
    %3 = firrtl.subfield %r[a]: !firrtl.bundle<a: uint<1>>
    firrtl.connect %2, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Simple combinational loop
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{y <- z <- y}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Combinational loop through a combinational memory read port
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{m.r.data <- m.r.en <- y <- z <- m.r.data}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %m_r = firrtl.mem Undefined  {depth = 2 : i64, name = "m", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %0 = firrtl.subfield %m_r[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %0, %clk : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %m_r[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %2 = firrtl.subfield %m_r[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %2, %y : !firrtl.uint<1>, !firrtl.uint<1>
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    firrtl.connect %2, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %3 = firrtl.subfield %m_r[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %z, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

// Combination loop through an instance
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  firrtl.module @thru1(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
  }

  firrtl.module @thru2(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %inner_in, %inner_out = firrtl.instance inner1 @thru1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inner_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{hasloops.{inner2.in <- y <- z <- inner2.out <- inner2.in}}}
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire  : !firrtl.uint<1>
    %z = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %inner_in, %inner_out = firrtl.instance inner2 @thru2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inner_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// CHECK: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"  {
  firrtl.module @thru1(in %clk: !firrtl.clock, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %reg = firrtl.reg  %clk  : !firrtl.clock, !firrtl.uint<1>
    firrtl.connect %reg, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %reg : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @thru2(in %clk: !firrtl.clock, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %inner1_clk, %inner1_in, %inner1_out = firrtl.instance inner1  @thru1(in clk: !firrtl.clock, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inner1_clk, %clk : !firrtl.clock, !firrtl.clock
    firrtl.connect %inner1_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %inner1_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire   : !firrtl.uint<1>
    %z = firrtl.wire   : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %inner2_clk, %inner2_in, %inner2_out = firrtl.instance inner2  @thru2(in clk: !firrtl.clock, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inner2_clk, %clk : !firrtl.clock, !firrtl.clock
    firrtl.connect %inner2_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %inner2_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "subaccess"   {
  // expected-error-re @below {{subaccess.{b[0].wo <- b[{{[0-3]}}].wo}}}
  firrtl.module @subaccess(in %sel1: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = firrtl.subindex %b[0] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>
    %3 = firrtl.subfield %2[wo]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    firrtl.matchingconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "subaccess"   {
  // expected-error-re @below {{subaccess.{b[{{[0-3]}}].wo <- b[{{[0-3]}}].wo}}}
  firrtl.module @subaccess(in %sel1: !firrtl.uint<2>, in %sel2: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = firrtl.subaccess %b[%sel2] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %3 = firrtl.subfield %2[wo]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    firrtl.matchingconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// CHECK: firrtl.circuit "subaccess"   {
firrtl.circuit "subaccess"   {
  firrtl.module @subaccess(in %sel1: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = firrtl.subindex %b[0] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>
    %3 = firrtl.subfield %2[wi]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    firrtl.matchingconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// CHECK: firrtl.circuit "subaccess"   {
firrtl.circuit "subaccess"   {
  firrtl.module @subaccess(in %sel1: !firrtl.uint<2>, in %sel2: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = firrtl.subaccess %b[%sel2] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %3 = firrtl.subfield %2[wi]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    firrtl.matchingconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// CHECK: firrtl.circuit "subaccess"   {
firrtl.circuit "subaccess"   {
  firrtl.module @subaccess(in %sel1: !firrtl.uint<2>, in %sel2: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = firrtl.subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %3 = firrtl.subfield %2[wi]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    firrtl.matchingconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// Two input ports share part of the path to an output port.
// CHECK-NOT: firrtl.circuit "revisitOps"
firrtl.circuit "revisitOps"   {
  firrtl.module @thru(in %in1: !firrtl.uint<1>,in %in2: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %1 = firrtl.mux(%in1, %in1, %in2)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{revisitOps.{inner2.in2 <- x <- inner2.out <- inner2.in2}}}
  firrtl.module @revisitOps() {
    %in1, %in2, %out = firrtl.instance inner2 @thru(in in1: !firrtl.uint<1>,in in2: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %in2, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %out : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Two input ports and a wire share path to an output port.
// CHECK-NOT: firrtl.circuit "revisitOps"
firrtl.circuit "revisitOps"   {
  firrtl.module @thru(in %in1: !firrtl.vector<uint<1>,2>, in %in2: !firrtl.vector<uint<1>,3>, out %out: !firrtl.vector<uint<1>,2>) {
    %w = firrtl.wire : !firrtl.uint<1>
    %in1_0 = firrtl.subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = firrtl.subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    %1 = firrtl.mux(%w, %in1_0, %in2_1)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out_1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{revisitOps.{inner2.in2[1] <- x <- inner2.out[1] <- inner2.in2[1]}}}
  firrtl.module @revisitOps() {
    %in1, %in2, %out = firrtl.instance inner2 @thru(in in1: !firrtl.vector<uint<1>,2>, in in2: !firrtl.vector<uint<1>,3>, out out: !firrtl.vector<uint<1>,2>)
    %in1_0 = firrtl.subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = firrtl.subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %in2_1, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %out_1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Shared comb path from input ports, ensure that all the paths to the output port are discovered.
// CHECK-NOT: firrtl.circuit "revisitOps"
firrtl.circuit "revisitOps"   {
  firrtl.module @thru(in %in0: !firrtl.vector<uint<1>,2>, in %in1: !firrtl.vector<uint<1>,2>, in %in2: !firrtl.vector<uint<1>,3>, out %out: !firrtl.vector<uint<1>,2>) {
    %w = firrtl.wire : !firrtl.uint<1>
    %in0_0 = firrtl.subindex %in0[0] : !firrtl.vector<uint<1>,2>
    %in1_0 = firrtl.subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = firrtl.subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    %1 = firrtl.mux(%w, %in1_0, %in2_1)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %2 = firrtl.mux(%w, %in0_0, %1)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out_1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{revisitOps.{inner2.in2[1] <- x <- inner2.out[1] <- inner2.in2[1]}}}
  firrtl.module @revisitOps() {
    %in0, %in1, %in2, %out = firrtl.instance inner2 @thru(in in0: !firrtl.vector<uint<1>,2>, in in1: !firrtl.vector<uint<1>,2>, in in2: !firrtl.vector<uint<1>,3>, out out: !firrtl.vector<uint<1>,2>)
    %in1_0 = firrtl.subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = firrtl.subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %in2_1, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %out_1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Comb path from ground type to aggregate.
// CHECK-NOT: firrtl.circuit "scalarToVec"
firrtl.circuit "scalarToVec"   {
  firrtl.module @thru(in %in1: !firrtl.uint<1>, in %in2: !firrtl.vector<uint<1>,3>, out %out: !firrtl.vector<uint<1>,2>) {
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    firrtl.connect %out_1, %in1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{scalarToVec.{inner2.in1 <- x <- inner2.out[1] <- inner2.in1}}}
  firrtl.module @scalarToVec() {
    %in1_0, %in2, %out = firrtl.instance inner2 @thru(in in1: !firrtl.uint<1>, in in2: !firrtl.vector<uint<1>,3>, out out: !firrtl.vector<uint<1>,2>)
    //%in1_0 = firrtl.subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %out_1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>,2>
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %in1_0, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %out_1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Check diagnostic produced if can't name anything on cycle.
// CHECK-NOT: firrtl.circuit "CycleWithoutNames"
firrtl.circuit "CycleWithoutNames"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, but unable to find names for any involved values.}}
  firrtl.module @CycleWithoutNames() {
    // expected-note @below {{cycle detected here}}
    %0 = firrtl.wire  : !firrtl.uint<1>
    firrtl.matchingconnect %0, %0 : !firrtl.uint<1>
  }
}

// -----

// Check diagnostic if starting point of detected cycle can't be named.
// Try to find something in the cycle we can name and start there.
firrtl.circuit "CycleStartsUnnammed"   {
  // expected-error @below {{sample path: CycleStartsUnnammed.{n <- ... <- n}}}
  firrtl.module @CycleStartsUnnammed() {
    %0 = firrtl.wire  : !firrtl.uint<1>
    %n = firrtl.node %0 : !firrtl.uint<1>
    firrtl.matchingconnect %0, %n : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "CycleThroughForceable"   {
  // expected-error @below {{sample path: CycleThroughForceable.{n <- w <- n}}}
  firrtl.module @CycleThroughForceable() {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    %n, %n_ref = firrtl.node %w forceable : !firrtl.uint<1>
    firrtl.matchingconnect %w, %n : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "CycleThroughForceableRef"   {
  // expected-error @below {{sample path: CycleThroughForceableRef.{n <- n <- w <- ... <- n}}}
  firrtl.module @CycleThroughForceableRef() {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    %n, %n_ref = firrtl.node %w forceable : !firrtl.uint<1>
    %read = firrtl.ref.resolve %n_ref : !firrtl.rwprobe<uint<1>>
    firrtl.matchingconnect %w, %read : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Force"   {
  // expected-error @below {{sample path: Force.{w <- w}}}
  firrtl.module @Force(in %clock: !firrtl.clock, in %x: !firrtl.uint<4>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.force %clock, %c, %w_ref, %w : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
  }
}

// -----

firrtl.circuit "RefDefineAndCastWidths" {
  firrtl.module @RefDefineAndCastWidths(in %x: !firrtl.uint<2>, out %p : !firrtl.probe<uint>) {
    %w, %ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %cast = firrtl.ref.cast %ref : (!firrtl.rwprobe<uint<2>>) -> !firrtl.probe<uint>
    firrtl.ref.define %p, %cast : !firrtl.probe<uint>
  }
}

// -----

firrtl.circuit "Properties"   {
  firrtl.module @Child(in %in: !firrtl.string, out %out: !firrtl.string) {
    firrtl.propassign %out, %in : !firrtl.string
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: Properties.{child0.in <- child0.out <- child0.in}}}
  firrtl.module @Properties() {
    %in, %out = firrtl.instance child0 @Child(in in: !firrtl.string, out out: !firrtl.string)
    firrtl.propassign %in, %out : !firrtl.string
  }
}

// -----

firrtl.circuit "hasnoloops"   {
  firrtl.module @thru(in %clk: !firrtl.clock, in %in1: !firrtl.uint<1>, in %in2: !firrtl.uint<1>, out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
    %a, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    firrtl.connect %out1, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.ref.force %clk, %a, %w_ref, %in1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
    firrtl.connect %out2, %in2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @hasnoloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %x = firrtl.wire  : !firrtl.uint<1>
    %clock, %inner_in1, %inner_in2, %inner_out1, %inner_out2 = firrtl.instance inner @thru(in clk: !firrtl.clock, in in1: !firrtl.uint<1>, in in2: !firrtl.uint<1>, out out1: !firrtl.uint<1>, out out2: !firrtl.uint<1>)
    firrtl.connect %inner_in1, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %inner_out1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %inner_in2, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %inner_out2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "forceLoop" {
  firrtl.module @thru1(in %clk: !firrtl.clock, in %in: !firrtl.uint<1>, out %out: !firrtl.rwprobe<uint<1>>) {
    %wire, %w_ref = firrtl.wire forceable :  !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %out, %w_ref  : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module @thru2(in %clk: !firrtl.clock, in %in: !firrtl.uint<1>, out %out: !firrtl.rwprobe<uint<1>>) {
    %inner1_clk, %inner1_in, %inner1_out = firrtl.instance inner1  @thru1(in clk: !firrtl.clock, in in: !firrtl.uint<1>, out out: !firrtl.rwprobe<uint<1>>)
    firrtl.connect %inner1_clk, %clk : !firrtl.clock, !firrtl.clock
    firrtl.connect %inner1_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.ref.define  %out, %inner1_out : !firrtl.rwprobe<uint<1>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: forceLoop.{inner2.in <- y <- z <- ... <- inner2.out <- inner2.in}}}
  firrtl.module @forceLoop(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = firrtl.wire   : !firrtl.uint<1>
    %z = firrtl.wire   : !firrtl.uint<1>
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %inner2_clk, %inner2_in, %w_ref = firrtl.instance inner2  @thru2(in clk: !firrtl.clock, in in: !firrtl.uint<1>, out out: !firrtl.rwprobe<uint<1>>)
    firrtl.connect %inner2_clk, %clk : !firrtl.clock, !firrtl.clock
    firrtl.connect %inner2_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.ref.force %clk, %c, %w_ref, %inner2_in : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
    %inner2_out = firrtl.ref.resolve %w_ref : !firrtl.rwprobe<uint<1>>
    firrtl.connect %z, %inner2_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Cycle through RWProbe ports.
firrtl.circuit "RefSink" {

  firrtl.module @RefSource(out %a_ref: !firrtl.probe<uint<1>>,
                           out %a_rwref: !firrtl.rwprobe<uint<1>>) {
    %a, %_a_rwref = firrtl.wire forceable : !firrtl.uint<1>,
      !firrtl.rwprobe<uint<1>>
    %b, %_b_rwref = firrtl.wire forceable : !firrtl.uint<1>,
      !firrtl.rwprobe<uint<1>>
    %a_ref_send = firrtl.ref.send %b : !firrtl.uint<1>
    firrtl.ref.define %a_ref, %a_ref_send : !firrtl.probe<uint<1>>
    firrtl.ref.define %a_rwref, %_a_rwref : !firrtl.rwprobe<uint<1>>
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }

// expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RefSink.{b <- ... <- refSource.a_ref <- refSource.a_rwref <- b}}}
  firrtl.module @RefSink(
    in %clock: !firrtl.clock,
    in %enable: !firrtl.uint<1>
  ) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %refSource_a_ref, %refSource_a_rwref =
      firrtl.instance refSource @RefSource(
        out a_ref: !firrtl.probe<uint<1>>,
        out a_rwref: !firrtl.rwprobe<uint<1>>
      )
    %a_ref_resolve =
      firrtl.ref.resolve %refSource_a_ref : !firrtl.probe<uint<1>>
    %b = firrtl.node %a_ref_resolve : !firrtl.uint<1>
    firrtl.ref.force_initial %c1_ui1, %refSource_a_rwref, %b :
      !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----
// Cycle through RWProbe ports.
firrtl.circuit "RefSink" {

  firrtl.module @RefSource(out %b_ref: !firrtl.rwprobe<uint<1>>,
                           out %a_rwref: !firrtl.rwprobe<uint<1>>) {
    %a, %_a_rwref = firrtl.wire forceable : !firrtl.uint<1>,
      !firrtl.rwprobe<uint<1>>
    %b, %_b_rwref = firrtl.wire forceable : !firrtl.uint<1>,
      !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %b_ref, %_b_rwref : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %a_rwref, %_a_rwref : !firrtl.rwprobe<uint<1>>
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }

// expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RefSink.{b <- ... <- refSource.b_ref <- refSource.a_rwref <- b}}}
  firrtl.module @RefSink(
    in %clock: !firrtl.clock,
    in %enable: !firrtl.uint<1>
  ) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %refSource_b_ref, %refSource_a_rwref =
      firrtl.instance refSource @RefSource(
        out b_ref: !firrtl.rwprobe<uint<1>>,
        out a_rwref: !firrtl.rwprobe<uint<1>>
      )
    %a_ref_resolve =
      firrtl.ref.resolve %refSource_b_ref : !firrtl.rwprobe<uint<1>>
    %b = firrtl.node %a_ref_resolve : !firrtl.uint<1>
    firrtl.ref.force_initial %c1_ui1, %refSource_a_rwref, %b :
      !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// Loop between two RWProbes referring to the same base value.
firrtl.circuit "RefSink" {
  firrtl.module @RefSource(out %a_rwref1: !firrtl.rwprobe<uint<1>>,
                           out %a_rwref2: !firrtl.rwprobe<uint<1>>) {
    %a, %_a_rwref = firrtl.wire forceable : !firrtl.uint<1>,
      !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %a_rwref1, %a_rwref2 : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %a_rwref2, %_a_rwref : !firrtl.rwprobe<uint<1>>
  }

// expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RefSink.{b <- ... <- refSource.a_rwref2 <- refSource.a_rwref1 <- b}}}
  firrtl.module @RefSink(
    in %clock: !firrtl.clock,
    in %enable: !firrtl.uint<1>
  ) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %refSource_a_rwref1, %refSource_a_rwref2 =
      firrtl.instance refSource @RefSource(
        out a_rwref1: !firrtl.rwprobe<uint<1>>,
        out a_rwref2: !firrtl.rwprobe<uint<1>>
      )
    %a_ref_resolve =
      firrtl.ref.resolve %refSource_a_rwref2 : !firrtl.rwprobe<uint<1>>
    %b = firrtl.node %a_ref_resolve : !firrtl.uint<1>
    firrtl.ref.force_initial %c1_ui1, %refSource_a_rwref1, %b :
      !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// Ensure deterministic error messages, in the presence of multiple probes.
firrtl.circuit "RefSink" {

  firrtl.module @RefSource(out %a_rwref1: !firrtl.rwprobe<uint<1>>,
                           out %a_rwref2: !firrtl.rwprobe<uint<1>>,
                           out %a_rwref3: !firrtl.rwprobe<uint<1>>,
                           out %a_rwref4: !firrtl.rwprobe<uint<1>>,
                           out %a_rwref5: !firrtl.rwprobe<uint<1>>) {
    %a, %_a_rwref = firrtl.wire forceable : !firrtl.uint<1>,
      !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %a_rwref1, %a_rwref2 : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %a_rwref2, %a_rwref3 : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %a_rwref3, %a_rwref4 : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %a_rwref4, %a_rwref5 : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %a_rwref5, %_a_rwref : !firrtl.rwprobe<uint<1>>
  }

// expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RefSink.{b <- ... <- refSource.a_rwref2 <- refSource.a_rwref1 <- b}}}
  firrtl.module @RefSink(
    in %clock: !firrtl.clock,
    in %enable: !firrtl.uint<1>
  ) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %rwref1, %rwref2, %rwref3, %rwref4, %rwref5 =
      firrtl.instance refSource @RefSource(
        out a_rwref1: !firrtl.rwprobe<uint<1>>,
        out a_rwref2: !firrtl.rwprobe<uint<1>>,
        out a_rwref3: !firrtl.rwprobe<uint<1>>,
        out a_rwref4: !firrtl.rwprobe<uint<1>>,
        out a_rwref5: !firrtl.rwprobe<uint<1>>
      )
    %a_ref_resolve =
      firrtl.ref.resolve %rwref2 : !firrtl.rwprobe<uint<1>>
    %b = firrtl.node %a_ref_resolve : !firrtl.uint<1>
    firrtl.ref.force_initial %c1_ui1, %rwref5, %b :
      !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// Incorrect visit of instance op results was resulting in missed cycles.
firrtl.circuit "Bug5442" {
  firrtl.module private @Bar(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
  firrtl.module private @Baz(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>, out %c_d: !firrtl.uint<1>) {
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
    firrtl.matchingconnect %c_d, %a : !firrtl.uint<1>
  }
// expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: Bug5442.{bar.a <- baz.b <- baz.a <- bar.b <- bar.a}}}
  firrtl.module @Bug5442() attributes {convention = #firrtl<convention scalarized>} {
    %bar_a, %bar_b = firrtl.instance bar @Bar(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %baz_a, %baz_b, %baz_c_d = firrtl.instance baz @Baz(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c_d: !firrtl.uint<1>)
    firrtl.matchingconnect %bar_a, %baz_b : !firrtl.uint<1>
    firrtl.matchingconnect %baz_a, %bar_b : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "RefSubLoop" {
  firrtl.module private @Child(in %bundle: !firrtl.bundle<a: uint<1>, b: uint<1>>, out %p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>) {
    %n, %n_ref = firrtl.node interesting_name %bundle forceable : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.ref.define %p, %n_ref : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RefSubLoop.{c.bundle.b <- ... <- c.p.b <- c.bundle.b}}}
  firrtl.module @RefSubLoop(in %x: !firrtl.uint<1>) {
    %c_bundle, %c_p = firrtl.instance c interesting_name @Child(in bundle: !firrtl.bundle<a: uint<1>, b: uint<1>>, out p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>)
    %0 = firrtl.ref.sub %c_p[1] : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    %1 = firrtl.subfield %c_bundle[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %2 = firrtl.subfield %c_bundle[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %2, %x : !firrtl.uint<1>
    %3 = firrtl.ref.resolve %0 : !firrtl.rwprobe<uint<1>>
    firrtl.matchingconnect %1, %3 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Issue4691" {
  firrtl.module private @Send(in %val: !firrtl.uint<2>, out %x: !firrtl.probe<uint<2>>) {
    %ref_val = firrtl.ref.send %val : !firrtl.uint<2>
    firrtl.ref.define %x, %ref_val : !firrtl.probe<uint<2>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: Issue4691.{sub.val <- ... <- sub.x <- sub.val}}}
  firrtl.module @Issue4691(out %x : !firrtl.uint<2>) {
    %sub_val, %sub_x = firrtl.instance sub @Send(in val: !firrtl.uint<2>, out x: !firrtl.probe<uint<2>>)
    %res = firrtl.ref.resolve %sub_x : !firrtl.probe<uint<2>>
    firrtl.connect %sub_val, %res : !firrtl.uint<2>, !firrtl.uint<2>
    firrtl.matchingconnect %x, %sub_val : !firrtl.uint<2>
  }
}

// -----

firrtl.circuit "Issue5462" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: Issue5462.{n.a <- w.a <- n.a}}}
  firrtl.module @Issue5462() attributes {convention = #firrtl<convention scalarized>} {
    %w = firrtl.wire : !firrtl.bundle<a: uint<8>>
    %n = firrtl.node %w : !firrtl.bundle<a: uint<8>>
    %0 = firrtl.subfield %n[a] : !firrtl.bundle<a: uint<8>>
    %1 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<8>>
    firrtl.matchingconnect %1, %0 : !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "Issue5462" {
  firrtl.module private @Child(in %bundle: !firrtl.bundle<a: uint<1>, b: uint<1>>, out %p: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %n = firrtl.node %bundle : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %0 = firrtl.subfield %n[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %1 = firrtl.subfield %p[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %1, %0 : !firrtl.uint<1>
    %2 = firrtl.subfield %n[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %3 = firrtl.subfield %p[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %3, %2 : !firrtl.uint<1>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: Issue5462.{c.bundle.b <- c.p.b <- c.bundle.b}}}
  firrtl.module @Issue5462(in %x: !firrtl.uint<1>) attributes {convention = #firrtl<convention scalarized>} {
    %c_bundle, %c_p = firrtl.instance c @Child(in bundle: !firrtl.bundle<a: uint<1>, b: uint<1>>, out p: !firrtl.bundle<a: uint<1>, b: uint<1>>)
    %0 = firrtl.subfield %c_p[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %1 = firrtl.subfield %c_bundle[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %2 = firrtl.subfield %c_bundle[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %2, %x : !firrtl.uint<1>
    firrtl.matchingconnect %1, %0 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Issue5462" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: Issue5462.{a <- n.a <- w.a <- a}}}
  firrtl.module @Issue5462(in %in_a: !firrtl.uint<8>, out %out_a: !firrtl.uint<8>, in %c: !firrtl.uint<1>) attributes {convention = #firrtl<convention scalarized>} {
    %w = firrtl.wire : !firrtl.bundle<a: uint<8>>
    %n = firrtl.node %w : !firrtl.bundle<a: uint<8>>
    %0 = firrtl.bundlecreate %in_a : (!firrtl.uint<8>) -> !firrtl.bundle<a: uint<8>>
    %1 = firrtl.mux(%c, %n, %0) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>) -> !firrtl.bundle<a: uint<8>>
    %2 = firrtl.subfield %1[a] : !firrtl.bundle<a: uint<8>>
    %3 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<8>>
    firrtl.matchingconnect %3, %2 : !firrtl.uint<8>
    %4 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<8>>
    firrtl.matchingconnect %out_a, %4 : !firrtl.uint<8>
  }
}

// -----
firrtl.circuit "FlipConnect1" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: FlipConnect1.{w.a <- x.a <- w.a}}}
  firrtl.module @FlipConnect1(){
    %w = firrtl.wire : !firrtl.bundle<a flip: uint<8>>
    %x = firrtl.wire : !firrtl.bundle<a flip: uint<8>>
    // x.a <= w.a
    firrtl.connect %w, %x: !firrtl.bundle<a flip: uint<8>>, !firrtl.bundle<a flip: uint<8>>
    // w.a <= x.a
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a flip: uint<8>>
    %x_a = firrtl.subfield %x[a] : !firrtl.bundle<a flip: uint<8>>
    firrtl.matchingconnect %w_a, %x_a : !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "FlipConnect2" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: FlipConnect2.{in.a.a <- out.a.a <- in.a.a}}}
  firrtl.module @FlipConnect2() {
    %in = firrtl.wire   : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %out  = firrtl.wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    firrtl.connect %out, %in : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>,
    !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %0 = firrtl.subfield %in[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %1 = firrtl.subfield %out[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    firrtl.connect %1, %0 : !firrtl.bundle<a flip: uint<1>>,!firrtl.bundle<a flip: uint<1>>
  }
}

// -----

firrtl.circuit "UnrealizedConversionCast" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: UnrealizedConversionCast.{b <- b <- b}}}
  firrtl.module @UnrealizedConversionCast() {
    // Casts have cast-like behavior
    %b = firrtl.wire   : !firrtl.uint<32>
    %a = builtin.unrealized_conversion_cast %b : !firrtl.uint<32> to !firrtl.uint<32>
    firrtl.matchingconnect %b, %a : !firrtl.uint<32>
  }
}

// -----

// Check RWProbeOp + force doesn't crash.
// No loop here.
firrtl.circuit "Issue6820" {
  firrtl.module private @Foo(in %x: !firrtl.uint<1> sym @sym, out %probe_bore: !firrtl.rwprobe<uint<1>>) {
    %0 = firrtl.ref.rwprobe <@Foo::@sym> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %probe_bore, %0 : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module @Issue6820(in %clock: !firrtl.clock, in %x: !firrtl.uint<1>, out %probe: !firrtl.rwprobe<uint<1>>) attributes {convention = #firrtl<convention scalarized>} {
    %foo_x, %foo_probe_bore = firrtl.instance foo @Foo(in x: !firrtl.uint<1>, out probe_bore: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %foo_x, %x : !firrtl.uint<1>
    firrtl.ref.define %probe, %foo_probe_bore : !firrtl.rwprobe<uint<1>>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1_ui1, %probe, %c0_ui1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// Single-element combinational loop under a when.
firrtl.circuit "WhenLoop" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: WhenLoop.{w <- w}}}
  firrtl.module @WhenLoop(in %in : !firrtl.uint<1>) {
    firrtl.when %in : !firrtl.uint<1> {
      %w = firrtl.wire  : !firrtl.uint<8>
      firrtl.connect %w, %w : !firrtl.uint<8>, !firrtl.uint<8>
    }
  }
}

// -----

// Single-element combinational loop under a match.
firrtl.circuit "MatchLoop" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: MatchLoop.{w <- w}}}
  firrtl.module @MatchLoop(in %in : !firrtl.enum<a: uint<1>>) {
    firrtl.match %in : !firrtl.enum<a: uint<1>> {
      case a(%arg0) {
        %w = firrtl.wire  : !firrtl.uint<8>
        firrtl.connect %w, %w : !firrtl.uint<8>, !firrtl.uint<8>
      }
    }
  }
}

// -----

// Single-element combinational loop under a layerblock.
firrtl.circuit "LayerLoop"   {
  firrtl.layer @A bind { }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: LayerLoop.{w <- w}}}
  firrtl.module @LayerLoop() {
    firrtl.layerblock @A {
      %w = firrtl.wire  : !firrtl.uint<8>
      firrtl.connect %w, %w : !firrtl.uint<8>, !firrtl.uint<8>
    }
  }
}

// -----

// Single-element combinational loop with multiple connects.
firrtl.circuit "MultipleConnects"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: MultipleConnects.{w <- w}}}
  firrtl.module @MultipleConnects(in %in : !firrtl.uint<8>) {
    %w = firrtl.wire  : !firrtl.uint<8>
    firrtl.connect %w, %w : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %w, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "DPI"   {
  firrtl.module @DPI(in %clock: !firrtl.clock, in %enable: !firrtl.uint<1>, out %out_0: !firrtl.uint<8>) attributes {convention = #firrtl<convention scalarized>} {
    %in = firrtl.wire : !firrtl.uint<8>
    %0 = firrtl.int.dpi.call "clocked_result"(%in) clock %clock enable %enable : (!firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.matchingconnect %in, %0 : !firrtl.uint<8>
    firrtl.matchingconnect %out_0, %0 : !firrtl.uint<8>
  }
}


// -----

firrtl.circuit "DPI"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: DPI.{in <- ... <- in}}}
  firrtl.module @DPI(in %clock: !firrtl.clock, in %enable: !firrtl.uint<1>, out %out_0: !firrtl.uint<8>) attributes {convention = #firrtl<convention scalarized>} {
    %in = firrtl.wire : !firrtl.uint<8>
    %0 = firrtl.int.dpi.call "unclocked_result"(%in) enable %enable : (!firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.matchingconnect %in, %0 : !firrtl.uint<8>
    firrtl.matchingconnect %out_0, %0 : !firrtl.uint<8>
  }
}

// -----

// Test that InstanceChoiceOp is handled correctly - no loop case
// CHECK: firrtl.circuit "InstanceChoiceNoLoop"
firrtl.circuit "InstanceChoiceNoLoop" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
    firrtl.option_case @ASIC
  }

  firrtl.module private @DefaultImpl(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<8>
  }

  firrtl.module private @FPGAImpl(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<8>
  }

  firrtl.module @InstanceChoiceNoLoop(in %x: !firrtl.uint<8>, out %y: !firrtl.uint<8>) {
    %inst_in, %inst_out = firrtl.instance_choice inst @DefaultImpl alternatives @Platform {
      @FPGA -> @FPGAImpl
    } (in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %inst_in, %x : !firrtl.uint<8>
    firrtl.matchingconnect %y, %inst_out : !firrtl.uint<8>
  }
}

// -----

// Test that InstanceChoiceOp detects loops through default module
firrtl.circuit "InstanceChoiceLoopDefault" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }

  firrtl.module private @LoopImpl(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<8>
  }

  firrtl.module private @NoLoopImpl(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %out, %c0_ui8 : !firrtl.uint<8>
  }

  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: InstanceChoiceLoopDefault.{inst.in <- y <- inst.out <- inst.in}}
  firrtl.module @InstanceChoiceLoopDefault(in %x: !firrtl.uint<8>) {
    %y = firrtl.wire : !firrtl.uint<8>
    %inst_in, %inst_out = firrtl.instance_choice inst @LoopImpl alternatives @Platform {
      @FPGA -> @NoLoopImpl
    } (in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %inst_in, %y : !firrtl.uint<8>
    firrtl.matchingconnect %y, %inst_out : !firrtl.uint<8>
  }
}

// -----

// Test that InstanceChoiceOp detects loops through alternative module
firrtl.circuit "InstanceChoiceLoopAlt" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }

  firrtl.module private @NoLoopImpl(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %out, %c0_ui8 : !firrtl.uint<8>
  }

  firrtl.module private @LoopImpl(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    firrtl.matchingconnect %out, %in : !firrtl.uint<8>
  }

  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: InstanceChoiceLoopAlt.{inst.in <- y <- inst.out <- inst.in}}
  firrtl.module @InstanceChoiceLoopAlt(in %x: !firrtl.uint<8>) {
    %y = firrtl.wire : !firrtl.uint<8>
    %inst_in, %inst_out = firrtl.instance_choice inst @NoLoopImpl alternatives @Platform {
      @FPGA -> @LoopImpl
    } (in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %inst_in, %y : !firrtl.uint<8>
    firrtl.matchingconnect %y, %inst_out : !firrtl.uint<8>
  }
}

// -----

// Forcing a register is not combinational, so forcing it from its own output
// is not a loop, in the same module or through an instance.
// CHECK: firrtl.circuit "ForceableRegInChild"
firrtl.circuit "ForceableRegInChild" {
  firrtl.module @Child(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>,
                       in %d: !firrtl.uint<8>, out %o: !firrtl.uint<8>,
                       out %p: !firrtl.rwprobe<uint<8>>) {
    %r, %r_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.matchingconnect %o, %r : !firrtl.uint<8>
    firrtl.ref.define %p, %r_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %en, %p, %o : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
  firrtl.module @ForceableRegInChild(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>,
                                     in %d: !firrtl.uint<8>, out %o: !firrtl.uint<8>) {
    %c_clock, %c_en, %c_d, %c_o, %c_p = firrtl.instance c @Child(in clock: !firrtl.clock, in en: !firrtl.uint<1>, in d: !firrtl.uint<8>, out o: !firrtl.uint<8>, out p: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %c_d, %d : !firrtl.uint<8>
    firrtl.matchingconnect %o, %c_o : !firrtl.uint<8>
    firrtl.ref.force %clock, %en, %c_p, %c_o : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Forcing a register through a RWProbeOp from its own output must not create a
// combinational loop.
// CHECK: firrtl.circuit "ForceRegisterDoesNotCreateCombLoop"
firrtl.circuit "ForceRegisterDoesNotCreateCombLoop" {
  firrtl.module @ForceRegisterDoesNotCreateCombLoop(
      in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
      in %enable: !firrtl.uint<1>, out %reg_out: !firrtl.uint<1>,
      out %regreset_out: !firrtl.uint<1>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %reg = firrtl.reg sym @reg %clock : !firrtl.clock, !firrtl.uint<1>
    %regreset = firrtl.regreset sym @regreset %clock, %reset, %zero : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    %reg_ref = firrtl.ref.rwprobe <@ForceRegisterDoesNotCreateCombLoop::@reg> : !firrtl.rwprobe<uint<1>>
    %regreset_ref = firrtl.ref.rwprobe <@ForceRegisterDoesNotCreateCombLoop::@regreset> : !firrtl.rwprobe<uint<1>>
    firrtl.matchingconnect %reg_out, %reg : !firrtl.uint<1>
    firrtl.matchingconnect %regreset_out, %regreset : !firrtl.uint<1>
    firrtl.ref.force %clock, %enable, %reg_ref, %reg_out : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
    firrtl.ref.force %clock, %enable, %regreset_ref, %regreset_out : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// Check simple RWProbeOp + read loop.
firrtl.circuit "RWProbeOpReadLoop" {
  firrtl.module private @Foo(in %x: !firrtl.uint<1> sym @sym,
                             out %probe_bore: !firrtl.rwprobe<uint<1>>) {
    %0 = firrtl.ref.rwprobe <@Foo::@sym> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %probe_bore, %0 : !firrtl.rwprobe<uint<1>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RWProbeOpReadLoop.{foo.probe_bore <- foo.x <- ... <- foo.probe_bore}}}
  firrtl.module @RWProbeOpReadLoop(in %x: !firrtl.uint<1>) {
    %foo_x, %foo_probe_bore = firrtl.instance foo @Foo(in x: !firrtl.uint<1>, out probe_bore: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %foo_x, %x : !firrtl.uint<1>
    %read = firrtl.ref.resolve %foo_probe_bore : !firrtl.rwprobe<uint<1>>
    firrtl.matchingconnect %foo_x, %read : !firrtl.uint<1>
  }
}

// -----

// Check RWProbeOp + force cycles are detected, when the force comes before the
// ref.define of the forced probe.
firrtl.circuit "RWProbeOpForceBeforeDefine" {
  firrtl.module private @Foo(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>, out %clockProbe_bore: !firrtl.rwprobe<uint<1>>) {
    %1 = firrtl.wire sym @sym: !firrtl.uint<1>
    firrtl.matchingconnect %1, %x: !firrtl.uint<1>
    firrtl.matchingconnect %y, %1: !firrtl.uint<1>
    %0 = firrtl.ref.rwprobe <@Foo::@sym> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %clockProbe_bore, %0 : !firrtl.rwprobe<uint<1>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RWProbeOpForceBeforeDefine.{foo.clockProbe_bore <- foo.y <- foo.clockProbe_bore}}}
  firrtl.module @RWProbeOpForceBeforeDefine(in %clock: !firrtl.clock, in %x: !firrtl.uint<1>, out %clockProbe: !firrtl.rwprobe<uint<1>>) {
    %foo_x, %foo_y, %foo_bore = firrtl.instance foo @Foo(in x: !firrtl.uint<1>, out y: !firrtl.uint<1>, out clockProbe_bore: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %foo_x, %x : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1_ui1, %clockProbe, %foo_y : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
    firrtl.ref.define %clockProbe, %foo_bore : !firrtl.rwprobe<uint<1>>
  }
}

// -----

// RWProbeOp on a field that does not drive the output, no loop.
// CHECK: firrtl.circuit "RWProbeOpSubfieldNoLoop"
firrtl.circuit "RWProbeOpSubfieldNoLoop" {
  firrtl.module private @Foo(in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out %y: !firrtl.uint<1>, out %p: !firrtl.rwprobe<uint<1>>) {
    %w = firrtl.wire sym [<@sym,2,public>] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %w, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %y, %w_a : !firrtl.uint<1>
    %0 = firrtl.ref.rwprobe <@Foo::@sym> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module @RWProbeOpSubfieldNoLoop(in %clock: !firrtl.clock, in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %foo_x, %foo_y, %foo_p = firrtl.instance foo @Foo(in x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out y: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %foo_x, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1_ui1, %foo_p, %foo_y : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// RWProbeOp on a field that drives the output, loop.
firrtl.circuit "RWProbeOpSubfieldLoop" {
  firrtl.module private @Foo(in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out %y: !firrtl.uint<1>, out %p: !firrtl.rwprobe<uint<1>>) {
    %w = firrtl.wire sym [<@sym,1,public>] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %w, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %y, %w_a : !firrtl.uint<1>
    %0 = firrtl.ref.rwprobe <@Foo::@sym> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint<1>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RWProbeOpSubfieldLoop.{foo.p <- foo.y <- foo.p}}}
  firrtl.module @RWProbeOpSubfieldLoop(in %clock: !firrtl.clock, in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %foo_x, %foo_y, %foo_p = firrtl.instance foo @Foo(in x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out y: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %foo_x, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1_ui1, %foo_p, %foo_y : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// RWProbeOp of an aggregate, with a loop only through the whole aggregate.
firrtl.circuit "RWProbeOpAggregateReadLoop" {
  firrtl.module private @Foo(in %x: !firrtl.bundle<a: uint<1>, b: uint<1>> sym @sym,
                             out %p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>) {
    %0 = firrtl.ref.rwprobe <@Foo::@sym> : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RWProbeOpAggregateReadLoop.{foo.p <- foo.x <- ... <- foo.p}}}
  firrtl.module @RWProbeOpAggregateReadLoop() {
    %foo_x, %foo_p = firrtl.instance foo @Foo(in x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>)
    %read = firrtl.ref.resolve %foo_p : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    %bits = firrtl.bitcast %read : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<2>
    %back = firrtl.bitcast %bits : (!firrtl.uint<2>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %foo_x, %back : !firrtl.bundle<a: uint<1>, b: uint<1>>
  }
}

// -----

// A cast RWProbe refers to the same data as its input.
firrtl.circuit "ForceThroughRefCast" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: ForceThroughRefCast.{w <- w}}}
  firrtl.module @ForceThroughRefCast(in %clock: !firrtl.clock, out %p: !firrtl.rwprobe<uint<1>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    %cast = firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<1>>) -> !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p, %cast : !firrtl.rwprobe<uint<1>>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1_ui1, %p, %w : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// A RWProbe of a field refers to the same field of the data.
firrtl.circuit "ForceThroughRefSub" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: ForceThroughRefSub.{w.a <- w.a}}}
  firrtl.module @ForceThroughRefSub(in %clock: !firrtl.clock, out %p: !firrtl.rwprobe<uint<1>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    %s = firrtl.ref.sub %w_ref[0] : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    firrtl.ref.define %p, %s : !firrtl.rwprobe<uint<1>>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1_ui1, %p, %w_a : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// A RWProbe port of a field drives the ports driven by that field.
firrtl.circuit "ForceExportedRefSub" {
  firrtl.module private @Child(in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>,
                               out %o: !firrtl.uint<1>,
                               out %p: !firrtl.rwprobe<uint<1>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    firrtl.matchingconnect %w, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %o, %w_a : !firrtl.uint<1>
    %s = firrtl.ref.sub %w_ref[0] : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    firrtl.ref.define %p, %s : !firrtl.rwprobe<uint<1>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: ForceExportedRefSub.{c.o <- c.p <- c.o}}}
  firrtl.module @ForceExportedRefSub(in %clock: !firrtl.clock, in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %c_x, %c_o, %c_p = firrtl.instance c @Child(in x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out o: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1_ui1, %c_p, %c_o : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// A RWProbe port of an output port. Forcing the RWProbe from the output port
// is a loop.
firrtl.circuit "RWProbeOfOutputPortForceLoop" {
  firrtl.module private @Child(in %x: !firrtl.uint<1>, out %o: !firrtl.uint<1> sym @o,
                               out %p: !firrtl.rwprobe<uint<1>>) {
    firrtl.matchingconnect %o, %x : !firrtl.uint<1>
    %0 = firrtl.ref.rwprobe <@Child::@o> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint<1>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RWProbeOfOutputPortForceLoop.{c.o <- c.o}}}
  firrtl.module @RWProbeOfOutputPortForceLoop(in %clock: !firrtl.clock, in %x: !firrtl.uint<1>) {
    %c_x, %c_o, %c_p = firrtl.instance c @Child(in x: !firrtl.uint<1>, out o: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.uint<1>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1, %c_p, %c_o : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// A RWProbe port of an output port has paths to and from it, but no loop.
// CHECK: firrtl.circuit "RWProbeOfOutputPortNoLoop"
firrtl.circuit "RWProbeOfOutputPortNoLoop" {
  firrtl.module private @Child(in %x: !firrtl.uint<1>, out %o: !firrtl.uint<1> sym @o,
                               out %p: !firrtl.rwprobe<uint<1>>) {
    firrtl.matchingconnect %o, %x : !firrtl.uint<1>
    %0 = firrtl.ref.rwprobe <@Child::@o> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module @RWProbeOfOutputPortNoLoop(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    %c_x, %c_o, %c_p = firrtl.instance c @Child(in x: !firrtl.uint<1>, out o: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.uint<1>
    firrtl.matchingconnect %y, %c_o : !firrtl.uint<1>
  }
}

// -----

// A RWProbe re-exported through an intermediate module. Forcing it from the
// output driven by the probed wire is a loop.
firrtl.circuit "ReexportedRWProbeForceLoop" {
  firrtl.module private @Child(in %x: !firrtl.uint<1>, out %o: !firrtl.uint<1>,
                               out %p: !firrtl.rwprobe<uint<1>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    firrtl.matchingconnect %w, %x : !firrtl.uint<1>
    firrtl.matchingconnect %o, %w : !firrtl.uint<1>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module private @Mid(in %x: !firrtl.uint<1>, out %o: !firrtl.uint<1>,
                             out %p: !firrtl.rwprobe<uint<1>>) {
    %c_x, %c_o, %c_p = firrtl.instance c @Child(in x: !firrtl.uint<1>, out o: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.uint<1>
    firrtl.matchingconnect %o, %c_o : !firrtl.uint<1>
    firrtl.ref.define %p, %c_p : !firrtl.rwprobe<uint<1>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: ReexportedRWProbeForceLoop.{m.o <- m.p <- m.o}}}
  firrtl.module @ReexportedRWProbeForceLoop(in %clock: !firrtl.clock, in %x: !firrtl.uint<1>) {
    %m_x, %m_o, %m_p = firrtl.instance m @Mid(in x: !firrtl.uint<1>, out o: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %m_x, %x : !firrtl.uint<1>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1, %m_p, %m_o : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// Force of an aggregate through a RWProbeOp, with a loop only through a field.
// force w = y, where y.b = w.b.
firrtl.circuit "RWProbeOpForceAggregateFieldLoop" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RWProbeOpForceAggregateFieldLoop.{w.b <- y.b <- w.b}}}
  firrtl.module @RWProbeOpForceAggregateFieldLoop(in %clock: !firrtl.clock, in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %w = firrtl.wire sym @w : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %w, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %y = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %w_b = firrtl.subfield %w[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %y_a = firrtl.subfield %y[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %y_b = firrtl.subfield %y[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %y_a, %w_b : !firrtl.uint<1>
    firrtl.matchingconnect %y_b, %w_b : !firrtl.uint<1>
    %p = firrtl.ref.rwprobe <@RWProbeOpForceAggregateFieldLoop::@w> : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1, %p, %y : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  }
}

// -----

// Force of a forceable aggregate wire, with a loop only through a field.
firrtl.circuit "ForceableForceAggregateFieldLoop" {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: ForceableForceAggregateFieldLoop.{w.b <- y.b <- w.b}}}
  firrtl.module @ForceableForceAggregateFieldLoop(in %clock: !firrtl.clock, in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    firrtl.matchingconnect %w, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %y = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %w_b = firrtl.subfield %w[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %y_a = firrtl.subfield %y[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %y_b = firrtl.subfield %y[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %y_a, %w_b : !firrtl.uint<1>
    firrtl.matchingconnect %y_b, %w_b : !firrtl.uint<1>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1, %w_ref, %y : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  }
}

// -----

// Force a field of an exported aggregate RWProbe port, from the output driven
// by the same field.
firrtl.circuit "RefSubOfRWProbePortForceLoop" {
  firrtl.module private @Child(in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>,
                               out %o: !firrtl.uint<1>,
                               out %p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    firrtl.matchingconnect %w, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.matchingconnect %o, %w_a : !firrtl.uint<1>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RefSubOfRWProbePortForceLoop.{c.o <- c.p.a <- c.o}}}
  firrtl.module @RefSubOfRWProbePortForceLoop(in %clock: !firrtl.clock, in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %c_x, %c_o, %c_p = firrtl.instance c @Child(in x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out o: !firrtl.uint<1>, out p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %s = firrtl.ref.sub %c_p[0] : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1, %s, %c_o : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// Forcing a register with a read of its own RWProbe is not a loop.
// CHECK: firrtl.circuit "ForceRegisterWithOwnProbeRead"
firrtl.circuit "ForceRegisterWithOwnProbeRead" {
  firrtl.module @ForceRegisterWithOwnProbeRead(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>,
                       in %d: !firrtl.uint<8>, out %p: !firrtl.rwprobe<uint<8>>) {
    %r, %r_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.ref.define %p, %r_ref : !firrtl.rwprobe<uint<8>>
    %rd = firrtl.ref.resolve %p : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %en, %p, %rd : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Same as above, through an instance.
// CHECK: firrtl.circuit "ForceRegisterPortWithOwnProbeRead"
firrtl.circuit "ForceRegisterPortWithOwnProbeRead" {
  firrtl.module private @Child(in %clock: !firrtl.clock, in %d: !firrtl.uint<8>, out %p: !firrtl.rwprobe<uint<8>>) {
    %r, %r_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.ref.define %p, %r_ref : !firrtl.rwprobe<uint<8>>
  }
  firrtl.module @ForceRegisterPortWithOwnProbeRead(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %c_clock, %c_d, %c_p = firrtl.instance c @Child(in clock: !firrtl.clock, in d: !firrtl.uint<8>, out p: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_d, %d : !firrtl.uint<8>
    %rd = firrtl.ref.resolve %c_p : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %en, %c_p, %rd : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Same as above, through an intermediate module.
// CHECK: firrtl.circuit "ForceReexportedRegisterProbe"
firrtl.circuit "ForceReexportedRegisterProbe" {
  firrtl.module private @Child(in %clock: !firrtl.clock, in %d: !firrtl.uint<8>, out %p: !firrtl.rwprobe<uint<8>>) {
    %r, %r_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.ref.define %p, %r_ref : !firrtl.rwprobe<uint<8>>
  }
  firrtl.module private @Mid(in %clock: !firrtl.clock, in %d: !firrtl.uint<8>, out %p: !firrtl.rwprobe<uint<8>>) {
    %c_clock, %c_d, %c_p = firrtl.instance c @Child(in clock: !firrtl.clock, in d: !firrtl.uint<8>, out p: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_d, %d : !firrtl.uint<8>
    firrtl.ref.define %p, %c_p : !firrtl.rwprobe<uint<8>>
  }
  firrtl.module @ForceReexportedRegisterProbe(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %m_clock, %m_d, %m_p = firrtl.instance m @Mid(in clock: !firrtl.clock, in d: !firrtl.uint<8>, out p: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %m_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %m_d, %d : !firrtl.uint<8>
    %rd = firrtl.ref.resolve %m_p : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %en, %m_p, %rd : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Same as above, for a field of the RWProbe port.
// CHECK: firrtl.circuit "ForceRefSubOfRegisterProbePort"
firrtl.circuit "ForceRefSubOfRegisterProbePort" {
  firrtl.module private @Child(in %clock: !firrtl.clock, in %d: !firrtl.bundle<a: uint<1>, b: uint<1>>, out %p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>) {
    %r, %r_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    firrtl.matchingconnect %r, %d : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.ref.define %p, %r_ref : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
  }
  firrtl.module @ForceRefSubOfRegisterProbePort(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %c_clock, %c_d, %c_p = firrtl.instance c @Child(in clock: !firrtl.clock, in d: !firrtl.bundle<a: uint<1>, b: uint<1>>, out p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_d, %d : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %s = firrtl.ref.sub %c_p[0] : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    %rd = firrtl.ref.resolve %s : !firrtl.rwprobe<uint<1>>
    firrtl.ref.force %clock, %en, %s, %rd : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// A RWProbe of a register in only one alternative, forcing it with its own
// read is a loop through the other alternative.
firrtl.circuit "InstanceChoiceRegisterProbeOneAltLoop" {
  firrtl.option @Opt { firrtl.option_case @B }
  firrtl.module private @A(in %clock: !firrtl.clock, in %d: !firrtl.uint<8>, out %p: !firrtl.rwprobe<uint<8>>) {
    %r, %r_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.ref.define %p, %r_ref : !firrtl.rwprobe<uint<8>>
  }
  firrtl.module private @B(in %clock: !firrtl.clock, in %d: !firrtl.uint<8>, out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %w, %d : !firrtl.uint<8>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: InstanceChoiceRegisterProbeOneAltLoop.{c.p <- ... <- c.p}}}
  firrtl.module @InstanceChoiceRegisterProbeOneAltLoop(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %c_clock, %c_d, %c_p = firrtl.instance_choice c @A alternatives @Opt { @B -> @B } (in clock: !firrtl.clock, in d: !firrtl.uint<8>, out p: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_d, %d : !firrtl.uint<8>
    %rd = firrtl.ref.resolve %c_p : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %en, %c_p, %rd : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// A RWProbe of a register in all the alternatives, no loop.
// CHECK: firrtl.circuit "InstanceChoiceRegisterProbeAllAlts"
firrtl.circuit "InstanceChoiceRegisterProbeAllAlts" {
  firrtl.option @Opt { firrtl.option_case @B }
  firrtl.module private @A(in %clock: !firrtl.clock, in %d: !firrtl.uint<8>, out %p: !firrtl.rwprobe<uint<8>>) {
    %r, %r_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.ref.define %p, %r_ref : !firrtl.rwprobe<uint<8>>
  }
  firrtl.module private @B(in %clock: !firrtl.clock, in %d: !firrtl.uint<8>, out %p: !firrtl.rwprobe<uint<8>>) {
    %r = firrtl.reg sym @r %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    %0 = firrtl.ref.rwprobe <@B::@r> : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint<8>>
  }
  firrtl.module @InstanceChoiceRegisterProbeAllAlts(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %c_clock, %c_d, %c_p = firrtl.instance_choice c @A alternatives @Opt { @B -> @B } (in clock: !firrtl.clock, in d: !firrtl.uint<8>, out p: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_d, %d : !firrtl.uint<8>
    %rd = firrtl.ref.resolve %c_p : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %en, %c_p, %rd : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Multiple RWProbe ports of an output port, no loop.
// CHECK: firrtl.circuit "MultipleRWProbesOfOutputPortNoLoop"
firrtl.circuit "MultipleRWProbesOfOutputPortNoLoop" {
  firrtl.module private @Child(in %x: !firrtl.uint<1>, out %o: !firrtl.uint<1> sym @o,
                               out %p1: !firrtl.rwprobe<uint<1>>, out %p2: !firrtl.rwprobe<uint<1>>) {
    firrtl.matchingconnect %o, %x : !firrtl.uint<1>
    %0 = firrtl.ref.rwprobe <@Child::@o> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p1, %0 : !firrtl.rwprobe<uint<1>>
    %1 = firrtl.ref.rwprobe <@Child::@o> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p2, %1 : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module @MultipleRWProbesOfOutputPortNoLoop(in %clock: !firrtl.clock, in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    %c_x, %c_o, %c_p1, %c_p2 = firrtl.instance c @Child(in x: !firrtl.uint<1>, out o: !firrtl.uint<1>, out p1: !firrtl.rwprobe<uint<1>>, out p2: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.uint<1>
    firrtl.matchingconnect %y, %c_o : !firrtl.uint<1>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1, %c_p2, %x : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// Multiple RWProbe ports of an output port. Forcing one of them from the output
// port is a loop.
firrtl.circuit "MultipleRWProbesOfOutputPortLoop" {
  firrtl.module private @Child(in %x: !firrtl.uint<1>, out %o: !firrtl.uint<1> sym @o,
                               out %p1: !firrtl.rwprobe<uint<1>>, out %p2: !firrtl.rwprobe<uint<1>>) {
    firrtl.matchingconnect %o, %x : !firrtl.uint<1>
    %0 = firrtl.ref.rwprobe <@Child::@o> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p1, %0 : !firrtl.rwprobe<uint<1>>
    %1 = firrtl.ref.rwprobe <@Child::@o> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p2, %1 : !firrtl.rwprobe<uint<1>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: MultipleRWProbesOfOutputPortLoop.{c.o <- ... <- c.o}}}
  firrtl.module @MultipleRWProbesOfOutputPortLoop(in %clock: !firrtl.clock, in %x: !firrtl.uint<1>) {
    %c_x, %c_o, %c_p1, %c_p2 = firrtl.instance c @Child(in x: !firrtl.uint<1>, out o: !firrtl.uint<1>, out p1: !firrtl.rwprobe<uint<1>>, out p2: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.uint<1>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    %n = firrtl.not %c_o : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.ref.force %clock, %c1, %c_p2, %n : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// A RWProbe port of an aggregate output port. Forcing a field from another
// field is not a loop.
// CHECK: firrtl.circuit "RWProbeOfAggregateOutputPortFieldNoLoop"
firrtl.circuit "RWProbeOfAggregateOutputPortFieldNoLoop" {
  firrtl.module private @Child(in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out %o: !firrtl.bundle<a: uint<1>, b: uint<1>> sym @o,
                               out %p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>) {
    firrtl.matchingconnect %o, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %0 = firrtl.ref.rwprobe <@Child::@o> : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
  }
  firrtl.module @RWProbeOfAggregateOutputPortFieldNoLoop(in %clock: !firrtl.clock, in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %c_x, %c_o, %c_p = firrtl.instance c @Child(in x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out o: !firrtl.bundle<a: uint<1>, b: uint<1>>, out p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    %s = firrtl.ref.sub %c_p[1] : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    %o_a = firrtl.subfield %c_o[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.ref.force %clock, %c1, %s, %o_a : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// A RWProbe port of an aggregate output port. Forcing a field from the same
// field is a loop.
firrtl.circuit "RWProbeOfAggregateOutputPortFieldLoop" {
  firrtl.module private @Child(in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out %o: !firrtl.bundle<a: uint<1>, b: uint<1>> sym @o,
                               out %p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>) {
    firrtl.matchingconnect %o, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %0 = firrtl.ref.rwprobe <@Child::@o> : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: RWProbeOfAggregateOutputPortFieldLoop.{c.o.b <- ... <- c.o.b}}}
  firrtl.module @RWProbeOfAggregateOutputPortFieldLoop(in %clock: !firrtl.clock, in %x: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %c_x, %c_o, %c_p = firrtl.instance c @Child(in x: !firrtl.bundle<a: uint<1>, b: uint<1>>, out o: !firrtl.bundle<a: uint<1>, b: uint<1>>, out p: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    %s = firrtl.ref.sub %c_p[1] : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>>>
    %o_b = firrtl.subfield %c_o[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %n = firrtl.not %o_b : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.ref.force %clock, %c1, %s, %n : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// A RWProbe of an output port, re-exported with the output port through an
// intermediate module. Forcing it from the output port is a loop.
firrtl.circuit "ReexportedRWProbeOfOutputPortLoop" {
  firrtl.module private @Child(in %x: !firrtl.uint<1>, out %o: !firrtl.uint<1> sym @o,
                               out %p: !firrtl.rwprobe<uint<1>>) {
    firrtl.matchingconnect %o, %x : !firrtl.uint<1>
    %0 = firrtl.ref.rwprobe <@Child::@o> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module private @Mid(in %x: !firrtl.uint<1>, out %o: !firrtl.uint<1>, out %p: !firrtl.rwprobe<uint<1>>) {
    %c_x, %c_o, %c_p = firrtl.instance c @Child(in x: !firrtl.uint<1>, out o: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.uint<1>
    firrtl.matchingconnect %o, %c_o : !firrtl.uint<1>
    firrtl.ref.define %p, %c_p : !firrtl.rwprobe<uint<1>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: ReexportedRWProbeOfOutputPortLoop.{m.o <- m.p <- m.o}}}
  firrtl.module @ReexportedRWProbeOfOutputPortLoop(in %clock: !firrtl.clock, in %x: !firrtl.uint<1>) {
    %m_x, %m_o, %m_p = firrtl.instance m @Mid(in x: !firrtl.uint<1>, out o: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %m_x, %x : !firrtl.uint<1>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1, %m_p, %m_o : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}

// -----

// A RWProbe of an output port, re-exported with the output port through an
// intermediate module, no loop.
// CHECK: firrtl.circuit "ReexportedRWProbeOfOutputPortNoLoop"
firrtl.circuit "ReexportedRWProbeOfOutputPortNoLoop" {
  firrtl.module private @Child(in %x: !firrtl.uint<1>, out %o: !firrtl.uint<1> sym @o,
                               out %p: !firrtl.rwprobe<uint<1>>) {
    firrtl.matchingconnect %o, %x : !firrtl.uint<1>
    %0 = firrtl.ref.rwprobe <@Child::@o> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module private @Mid(in %x: !firrtl.uint<1>, out %o: !firrtl.uint<1>, out %p: !firrtl.rwprobe<uint<1>>) {
    %c_x, %c_o, %c_p = firrtl.instance c @Child(in x: !firrtl.uint<1>, out o: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %c_x, %x : !firrtl.uint<1>
    firrtl.matchingconnect %o, %c_o : !firrtl.uint<1>
    firrtl.ref.define %p, %c_p : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module @ReexportedRWProbeOfOutputPortNoLoop(in %clock: !firrtl.clock, in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    %m_x, %m_o, %m_p = firrtl.instance m @Mid(in x: !firrtl.uint<1>, out o: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<1>>)
    firrtl.matchingconnect %m_x, %x : !firrtl.uint<1>
    firrtl.matchingconnect %y, %m_o : !firrtl.uint<1>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1, %m_p, %x : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.uint<1>
  }
}
