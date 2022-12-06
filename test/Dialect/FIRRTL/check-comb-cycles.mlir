// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-comb-loops))' --split-input-file --verify-diagnostics %s | FileCheck %s

module  {
  // Loop-free circuit
  // CHECK: firrtl.circuit "hasnoloops"
  firrtl.circuit "hasnoloops"   {
    firrtl.module @thru(in %in1: !firrtl.uint<1>, in %in2: !firrtl.uint<1>, out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
      firrtl.connect %out1, %in1 : !firrtl.uint<1>, !firrtl.uint<1>
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
}

// -----

module  {
  // Simple combinational loop
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module, sample path: hasloops.y <- hasloops.z <- hasloops.y}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %z, %y : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}

// -----

module  {
  // Single-element combinational loop
  // CHECK-NOT: firrtl.circuit "loop"
  firrtl.circuit "loop"   {
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module, sample path: loop.w <- loop.w}}
    firrtl.module @loop(out %y: !firrtl.uint<8>) {
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %w = firrtl.wire  : !firrtl.uint<8>
      firrtl.connect %w, %w : !firrtl.uint<8>, !firrtl.uint<8>
    }
  }
}

// -----

module  {
  // Node combinational loop
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+2 {{detected combinational cycle in a FIRRTL module, sample path: hasloops.y <- hasloops.z <-}}
    // expected-remark @+1 {{this operation is part of the combinational cycle, module argument 'b'}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      %y = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %0 = firrtl.and %c, %y : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      %z = firrtl.node %0  : !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}

// -----

module  {
  // Combinational loop through a combinational memory read port
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module, sample path: hasloops.y <- hasloops.z <- hasloops.m.r.data <- hasloops.m.r.addr <- hasloops.y <-}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{memory is part of a combinational cycle}}
      %m_r = firrtl.mem Undefined  {depth = 2 : i64, name = "m", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
      %0 = firrtl.subfield %m_r[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
      firrtl.connect %0, %clk : !firrtl.clock, !firrtl.clock
      %1 = firrtl.subfield %m_r(0) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %1, %y : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %m_r[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
      %c1_ui = firrtl.constant 1 : !firrtl.uint
      firrtl.connect %2, %c1_ui : !firrtl.uint<1>, !firrtl.uint
      %3 = firrtl.subfield %m_r(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %z, %3 : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}

// -----

module  {
  // Combination loop through an instance
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    firrtl.module @thru(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
      firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
    }
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module, sample path: hasloops.y <- hasloops.z <- hasloops.inner.out <- hasloops.inner.in <- hasloops.y <-}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{instance is part of a combinational cycle, instance port number '1' has a path from port number '0', inner.out <- inner.in}}
      %inner_in, %inner_out = firrtl.instance inner @thru(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
      firrtl.connect %inner_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %z, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}

// -----

module  {
  // Multiple simple loops in one SCC
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+2 {{hasloops.d <- hasloops.e <- hasloops.b <-}}
    // expected-remark @+1 {{this operation is part of the combinational cycle, module argument 'i'}}
    firrtl.module @hasloops(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
      %a = firrtl.wire  : !firrtl.uint<1>
      %b = firrtl.wire  : !firrtl.uint<1>
      %c = firrtl.wire  : !firrtl.uint<1>
      %d = firrtl.wire  : !firrtl.uint<1>
      %e = firrtl.wire  : !firrtl.uint<1>
      %0 = firrtl.and %c, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %1 = firrtl.and %a, %d : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %2 = firrtl.and %c, %e : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %d, %2 : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %e, %b : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %o, %e : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}

// -----

firrtl.circuit "strictConnectAndConnect" {
  // expected-error @+2 {{strictConnectAndConnect.a <- strictConnectAndConnect.b <- strictConnectAndConnect.a <-}}
  // expected-remark @+1 {{this operation is part of the combinational cycle}}
  firrtl.module @strictConnectAndConnect(out %a: !firrtl.uint<11>, out %b: !firrtl.uint<11>) {
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.connect %a, %b : !firrtl.uint<11>, !firrtl.uint<11>
    firrtl.strictconnect %b, %a : !firrtl.uint<11>
  }
}

// -----

firrtl.circuit "vectorRegInit"   {
  firrtl.module @vectorRegInit(in %clk: !firrtl.clock) {
    %reg = firrtl.reg %clk : !firrtl.vector<uint<8>, 2>
    %0 = firrtl.subindex %reg[0] : !firrtl.vector<uint<8>, 2>
    firrtl.connect %0, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "bundleRegInit"   {
  firrtl.module @bundleRegInit(in %clk: !firrtl.clock) {
    %reg = firrtl.reg %clk : !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %reg[a] : !firrtl.bundle<a: uint<1>>
    firrtl.connect %0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo"  {
  firrtl.extmodule private @Bar(in a: !firrtl.uint<1>)
  // expected-error @+2 {{Foo.a <- Foo.bar.a <- Foo.a <-}}
  // expected-remark @+1 {{this operation is part of the combinational cycle}}
  firrtl.module @Foo(out %a: !firrtl.uint<1>) {
    %bar_a = firrtl.instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %a : !firrtl.uint<1>
    // expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.strictconnect %a, %bar_a : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo"  {
  firrtl.module private @Bar(in %a: !firrtl.uint<1>) {}
  // expected-error @+2 {{detected combinational cycle in a FIRRTL module, sample path: Foo.a <- Foo.bar.a <- Foo.a}}
  // expected-remark @+1 {{this operation is part of the combinational cycle, module argument 'a'}}
  firrtl.module @Foo(out %a: !firrtl.uint<1>) {
    %bar_a = firrtl.instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %a : !firrtl.uint<1>
    // expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.strictconnect %a, %bar_a : !firrtl.uint<1>
  }
}

// -----

module  {
  // Node combinational loop through vector subindex
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+2 {{hasloops.w[3] <- hasloops.z <-}}
    // expected-remark @+1 {{this operation is part of the combinational cycle, module argument 'b'}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      %w = firrtl.wire  : !firrtl.vector<uint<1>,10>
      %y = firrtl.subindex %w[3]  : !firrtl.vector<uint<1>,10>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %0 = firrtl.and %c, %y : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      %z = firrtl.node %0  : !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}

// -----

  // Node combinational loop through vector subindex
  // CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @+2 {{detected combinational cycle in a FIRRTL module, sample path: hasloops.b[0] <- hasloops.bar_b[0] <- hasloops.bar_a[0] <- hasloops.b[0]}}
	// expected-remark @+1 {{this operation is part of the combinational cycle, module argument 'b[0]'}}
  firrtl.module @hasloops(out %b: !firrtl.vector<uint<1>, 2>) {
    %bar_a = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %bar_b = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
		// expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %4 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
		// expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.strictconnect %5, %4 : !firrtl.uint<1>
    %v0 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    %v1 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
		// expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.strictconnect %v1, %v0 : !firrtl.uint<1>
  }
}

// -----

// Combinational loop through instance ports
// CHECK-NOT: firrtl.circuit "hasloops"
firrtl.circuit "hasLoops"  {
  // expected-error @+2 {{detected combinational cycle in a FIRRTL module, sample path: hasLoops.b[0] <- hasLoops.bar.b[0] <- hasLoops.bar.a[0] <- hasLoops.b[0]}}
  // expected-remark @+1 {{this operation is part of the combinational cycle}}
  firrtl.module @hasLoops(out %b: !firrtl.vector<uint<1>, 2>) {
		// expected-remark @+1 {{instance is part of a combinational cycle, instance port number '1' has a path from port number '0', bar.b[0] <- bar.a[0]}}
    %bar_a, %bar_b = firrtl.instance bar  @Bar(in a: !firrtl.vector<uint<1>, 2>, out b: !firrtl.vector<uint<1>, 2>)
    %0 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
		// expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
    %2 = firrtl.subindex %b[1] : !firrtl.vector<uint<1>, 2>
    %3 = firrtl.subindex %bar_a[1] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %3, %2 : !firrtl.uint<1>
    %4 = firrtl.subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
		// expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.strictconnect %5, %4 : !firrtl.uint<1>
    %6 = firrtl.subindex %bar_b[1] : !firrtl.vector<uint<1>, 2>
    %7 = firrtl.subindex %b[1] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %7, %6 : !firrtl.uint<1>
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

  // expected-error @+2 {{detected combinational cycle in a FIRRTL module, sample path: bundleWire.w.foo.bar.baz <- bundleWire.out2 <- bundleWire.x <- bundleWire.w.foo.bar.baz}}
  // expected-remark @+1 {{this operation is part of the combinational cycle}}
  firrtl.module @bundleWire(in %arg: !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>,
                           out %out1: !firrtl.uint<1>, out %out2: !firrtl.sint<64>) {

    %w = firrtl.wire : !firrtl.bundle<foo: bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>>
    %w0 = firrtl.subfield %w(0) : (!firrtl.bundle<foo: bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>>) -> !firrtl.bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>
    %w0_0 = firrtl.subfield %w0(0) : (!firrtl.bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>) -> !firrtl.bundle<baz: sint<64>>
    %w0_0_0 = firrtl.subfield %w0_0(0) : (!firrtl.bundle<baz: sint<64>>) -> !firrtl.sint<64>
    %x = firrtl.wire  : !firrtl.sint<64>

    %0 = firrtl.subfield %arg(0) : (!firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>) -> !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    %1 = firrtl.subfield %0(0) : (!firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>) -> !firrtl.bundle<baz: uint<1>>
    %2 = firrtl.subfield %1(0) : (!firrtl.bundle<baz: uint<1>>) -> !firrtl.uint<1>
    %3 = firrtl.subfield %0(1) : (!firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>) -> !firrtl.sint<64>
    firrtl.connect %w0_0_0, %3 : !firrtl.sint<64>, !firrtl.sint<64>
		// expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.connect %x, %w0_0_0 : !firrtl.sint<64>, !firrtl.sint<64>
		// expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.connect %out2, %x : !firrtl.sint<64>, !firrtl.sint<64>
		// expected-remark @+1 {{this operation is part of the combinational cycle}}
    firrtl.connect %w0_0_0, %out2 : !firrtl.sint<64>, !firrtl.sint<64>

    firrtl.connect %out1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "registerLoop"   {
  // CHECK: firrtl.module @registerLoop(in %clk: !firrtl.clock)
  firrtl.module @registerLoop(in %clk: !firrtl.clock) {
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %r = firrtl.reg %clk : !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %w(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %w(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %r(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    %3 = firrtl.subfield %r(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %2, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

module  {
  // Simple combinational loop
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+1 {{hasloops.y <- hasloops.z <- hasloops.y <-}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %z, %y : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}

// -----

module  {
  // Combinational loop through a combinational memory read port
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+1 {{hasloops.y <- hasloops.z <- hasloops.m.r.data <- hasloops.m.r.en <- hasloops.y <-}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{memory is part of a combinational cycle}}
      %m_r = firrtl.mem Undefined  {depth = 2 : i64, name = "m", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
      %0 = firrtl.subfield %m_r(2) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.clock
      firrtl.connect %0, %clk : !firrtl.clock, !firrtl.clock
      %1 = firrtl.subfield %m_r(0) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
      %2 = firrtl.subfield %m_r(1) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %2, %y : !firrtl.uint<1>, !firrtl.uint<1>
      %c1_ui = firrtl.constant 1 : !firrtl.uint
      firrtl.connect %2, %c1_ui : !firrtl.uint<1>, !firrtl.uint
      %3 = firrtl.subfield %m_r(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %z, %3 : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}

// -----

module  {
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
    // expected-error @+1 {{hasloops.y <- hasloops.z <- hasloops.inner2.out <- hasloops.inner2.in <- hasloops.y <-}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{instance is part of a combinational cycle, instance port number '1' has a path from port number '0', inner2.out <- inner2.in}}
      %inner_in, %inner_out = firrtl.instance inner2 @thru2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
      firrtl.connect %inner_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %z, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}
