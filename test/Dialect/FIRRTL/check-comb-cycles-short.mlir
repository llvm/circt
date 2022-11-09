// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-comb-cycles))' --split-input-file --verify-diagnostics %s | FileCheck %s

module  {
  // Simple combinational loop
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-note @+1 {{y}}
      %y = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{z}}
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
  // Combinational loop through a combinational memory read port
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-note @+1 {{hasloops.y}}
      %y = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{hasloops.z}}
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      %m_r = firrtl.mem Undefined  {depth = 2 : i64, name = "m", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
      %0 = firrtl.subfield %m_r(2) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.clock
      firrtl.connect %0, %clk : !firrtl.clock, !firrtl.clock
      // expected-note @+1 {{hasloops.m.r.addr}}
      %1 = firrtl.subfield %m_r(0) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %1, %y : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %m_r(1) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
      %c1_ui = firrtl.constant 1 : !firrtl.uint
      firrtl.connect %2, %c1_ui : !firrtl.uint<1>, !firrtl.uint
      // expected-note @+1 {{hasloops.m.r.data}}
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
    // expected-note @+2 {{hasloops.inner2.inner1.in}}
    // expected-note @+1 {{hasloops.inner2.inner1.out}}
    firrtl.module @thru1(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
      firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
    }

    // expected-note @+2 {{hasloops.inner2.in}}
    // expected-note @+1 {{hasloops.inner2.out}}
    firrtl.module @thru2(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
      // expected-note @+1 {{Instance hasloops.inner2.inner1: in ---> out}}
      %inner_in, %inner_out = firrtl.instance inner1 @thru1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
      firrtl.connect %inner_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %out, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
    }
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-note @+1 {{hasloops.y}}
      %y = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{hasloops.z}}
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-note @+2 {{Instance hasloops.inner2: in ---> out}}
      // expected-note @+1 {{hasloops.inner2.in}}
      %inner_in, %inner_out = firrtl.instance inner2 @thru2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
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
      %a = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{hasloops.b}}
      %b = firrtl.wire  : !firrtl.uint<1>
      %c = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{hasloops.d}}
      %d = firrtl.wire  : !firrtl.uint<1>
      // expected-note @+1 {{hasloops.e}}
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
}