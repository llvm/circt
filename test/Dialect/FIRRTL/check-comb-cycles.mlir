// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-comb-loops))' --split-input-file  --verify-diagnostics %s | FileCheck %s

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

firrtl.circuit "strictConnectAndConnect" {
  // expected-error @below {{strictConnectAndConnect.{a <- b <- a}}}
  firrtl.module @strictConnectAndConnect(out %a: !firrtl.uint<11>, out %b: !firrtl.uint<11>) {
    %w = firrtl.wire : !firrtl.uint<11>
    firrtl.strictconnect %b, %w : !firrtl.uint<11>
    firrtl.connect %a, %b : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.strictconnect %b, %a : !firrtl.uint<11>
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
    firrtl.strictconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.strictconnect %a, %bar_a : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo"  {
  firrtl.module private @Bar(in %a: !firrtl.uint<1>) {}
  // expected-error @below {{Foo.{a <- bar.a <- a}}}
  firrtl.module @Foo(out %a: !firrtl.uint<1>) {
    %bar_a = firrtl.instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.strictconnect %a, %bar_a : !firrtl.uint<1>
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
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %4 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %5, %4 : !firrtl.uint<1>
    %v0 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    %v1 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %v1, %v0 : !firrtl.uint<1>
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
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %4 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %5, %4 : !firrtl.uint<1>
  }
   
  firrtl.module private @Bar(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %a[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %2 = firrtl.subindex %a[1] : !firrtl.vector<uint<1>, 2>
    %3 = firrtl.subindex %b[1] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %3, %2 : !firrtl.uint<1>
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
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %4 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %5, %4 : !firrtl.uint<1>
  }
   
  firrtl.module private @Bar(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    firrtl.strictconnect %b, %a : !firrtl.vector<uint<1>, 2>
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
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
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
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
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
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
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
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
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
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
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
    firrtl.strictconnect %0, %0 : !firrtl.uint<1>
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
    firrtl.strictconnect %0, %n : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "CycleThroughForceable"   {
  // expected-error @below {{sample path: CycleThroughForceable.{n <- w <- n}}}
  firrtl.module @CycleThroughForceable() {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    %n, %n_ref = firrtl.node %w forceable : !firrtl.uint<1>
    firrtl.strictconnect %w, %n : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "CycleThroughForceableRef"   {
  // expected-error @below {{sample path: CycleThroughForceableRef.{n <- n <- w <- ... <- n}}}
  firrtl.module @CycleThroughForceableRef() {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    %n, %n_ref = firrtl.node %w forceable : !firrtl.uint<1>
    %read = firrtl.ref.resolve %n_ref : !firrtl.rwprobe<uint<1>>
    firrtl.strictconnect %w, %read : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Force"   {
  // expected-error @below {{sample path: Force.{w <- w}}}
  firrtl.module @Force(in %clock: !firrtl.clock, in %x: !firrtl.uint<4>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.force %clock, %c, %w_ref, %w : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>
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
    firrtl.ref.force %clk, %a, %w_ref, %in1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
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
    firrtl.ref.force %clk, %c, %w_ref, %inner2_in : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
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
    firrtl.strictconnect %b, %a : !firrtl.uint<1>
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
      !firrtl.uint<1>, !firrtl.uint<1>
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
    firrtl.strictconnect %b, %a : !firrtl.uint<1>
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
      !firrtl.uint<1>, !firrtl.uint<1>
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
      !firrtl.uint<1>, !firrtl.uint<1>
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
      !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Incorrect visit of instance op results was resulting in missed cycles.
firrtl.circuit "Bug5442" {
  firrtl.module private @Bar(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    firrtl.strictconnect %b, %a : !firrtl.uint<1>
  }
  firrtl.module private @Baz(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>, out %c_d: !firrtl.uint<1>) {
    firrtl.strictconnect %b, %a : !firrtl.uint<1>
    firrtl.strictconnect %c_d, %a : !firrtl.uint<1>
  }
// expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: Bug5442.{bar.a <- baz.b <- baz.a <- bar.b <- bar.a}}}
  firrtl.module @Bug5442() attributes {convention = #firrtl<convention scalarized>} {
    %bar_a, %bar_b = firrtl.instance bar @Bar(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %baz_a, %baz_b, %baz_c_d = firrtl.instance baz @Baz(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c_d: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %baz_b : !firrtl.uint<1>
    firrtl.strictconnect %baz_a, %bar_b : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "References"   {
  firrtl.module private @Child(in %in: !firrtl.probe<uint<1>>, out %out: !firrtl.probe<uint<1>>) {
    firrtl.ref.define %out, %in : !firrtl.probe<uint<1>>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: References.{child0.in <- child0.out <- child0.in}}}
  firrtl.module @References() {
    %in, %out = firrtl.instance child0 @Child(in in: !firrtl.probe<uint<1>>, out out: !firrtl.probe<uint<1>>)
    firrtl.ref.define %in, %out : !firrtl.probe<uint<1>>
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
    firrtl.strictconnect %2, %x : !firrtl.uint<1>
    %3 = firrtl.ref.resolve %0 : !firrtl.rwprobe<uint<1>>
    firrtl.strictconnect %1, %3 : !firrtl.uint<1>
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
    firrtl.strictconnect %x, %sub_val : !firrtl.uint<2>
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
    firrtl.strictconnect %1, %0 : !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "Issue5462" {
  firrtl.module private @Child(in %bundle: !firrtl.bundle<a: uint<1>, b: uint<1>>, out %p: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    %n = firrtl.node %bundle : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %0 = firrtl.subfield %n[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %1 = firrtl.subfield %p[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %2 = firrtl.subfield %n[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %3 = firrtl.subfield %p[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.strictconnect %3, %2 : !firrtl.uint<1>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: Issue5462.{c.bundle.b <- c.p.b <- c.bundle.b}}}
  firrtl.module @Issue5462(in %x: !firrtl.uint<1>) attributes {convention = #firrtl<convention scalarized>} {
    %c_bundle, %c_p = firrtl.instance c @Child(in bundle: !firrtl.bundle<a: uint<1>, b: uint<1>>, out p: !firrtl.bundle<a: uint<1>, b: uint<1>>)
    %0 = firrtl.subfield %c_p[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %1 = firrtl.subfield %c_bundle[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %2 = firrtl.subfield %c_bundle[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.strictconnect %2, %x : !firrtl.uint<1>
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
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
    firrtl.strictconnect %3, %2 : !firrtl.uint<8>
    %4 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<8>>
    firrtl.strictconnect %out_a, %4 : !firrtl.uint<8>
  }
}
