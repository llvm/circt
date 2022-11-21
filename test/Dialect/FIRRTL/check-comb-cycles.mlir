// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-comb-cycles{print-simple-cycle=false}))' --split-input-file --verify-diagnostics %s | FileCheck %s

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
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %z, %y : !firrtl.uint<1>, !firrtl.uint<1>
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
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @loop(out %y: !firrtl.uint<8>) {
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %w = firrtl.wire  : !firrtl.uint<8>
      firrtl.connect %w, %w : !firrtl.uint<8>, !firrtl.uint<8>
      firrtl.connect %y, %w : !firrtl.uint<8>, !firrtl.uint<8>
    }
  }
}

// -----

module  {
  // Node combinational loop
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %0 = firrtl.and %c, %y : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %z = firrtl.node %0  : !firrtl.uint<1>
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
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      %m_r = firrtl.mem Undefined  {depth = 2 : i64, name = "m", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
      %0 = firrtl.subfield %m_r(2) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.clock
      firrtl.connect %0, %clk : !firrtl.clock, !firrtl.clock
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %1 = firrtl.subfield %m_r(0) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %1, %y : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %m_r(1) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
      %c1_ui = firrtl.constant 1 : !firrtl.uint
      firrtl.connect %2, %c1_ui : !firrtl.uint<1>, !firrtl.uint
      %3 = firrtl.subfield %m_r(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %z, %3 : !firrtl.uint<1>, !firrtl.uint<1>
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
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %inner_in, %inner_out = firrtl.instance inner @thru(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
      firrtl.connect %inner_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %z, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
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
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %a = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %b = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %c = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %d = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %e = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %0 = firrtl.and %c, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %1 = firrtl.and %a, %d : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-note @+1 {{this operation is part of the combinational cycle}}
      %2 = firrtl.and %c, %e : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %d, %2 : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %e, %b : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %o, %e : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}

// -----

firrtl.circuit "strictConnectAndConnect" {
  // expected-error @+2 {{detected combinational cycle in a FIRRTL module}}
  // expected-note @+1 {{this operation is part of the combinational cycle}}
  firrtl.module @strictConnectAndConnect(out %a: !firrtl.uint<11>, out %b: !firrtl.uint<11>) {
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
    %0 = firrtl.subfield %reg(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo"  {
  firrtl.extmodule private @Bar(in a: !firrtl.uint<1>)
  // expected-error @+2 {{detected combinational cycle in a FIRRTL module}}
  // expected-note @+1 {{this operation is part of the combinational cycle}}
  firrtl.module @Foo(out %a: !firrtl.uint<1>) {
    // expected-note @+1 {{this operation is part of the combinational cycle}}
    %bar_a = firrtl.instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.strictconnect %a, %bar_a : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo"  {
  firrtl.module private @Bar(in %a: !firrtl.uint<1>) {}
  // expected-error @+2 {{detected combinational cycle in a FIRRTL module}}
  // expected-note @+1 {{this operation is part of the combinational cycle}}
  firrtl.module @Foo(out %a: !firrtl.uint<1>) {
    // expected-note @+1 {{this operation is part of the combinational cycle}}
    %bar_a = firrtl.instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.strictconnect %a, %bar_a : !firrtl.uint<1>
  }
}
