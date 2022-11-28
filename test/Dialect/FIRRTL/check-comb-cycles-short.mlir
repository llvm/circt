// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-comb-loops))' --split-input-file --verify-diagnostics %s | FileCheck %s

module  {
  // Simple combinational loop
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %y = firrtl.wire  : !firrtl.uint<1>
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
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
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      %y = firrtl.wire  : !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{memory is part of a combinational cycle between the data and address port}}
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
    firrtl.module @thru1(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
      firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
    }

    firrtl.module @thru2(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
      %inner_in, %inner_out = firrtl.instance inner1 @thru1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
      firrtl.connect %inner_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %out, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
    }
    // expected-error @+1 {{detected combinational cycle in a FIRRTL module}}
    firrtl.module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
      %y = firrtl.wire  : !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %z = firrtl.wire  : !firrtl.uint<1>
      firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{instance is part of a combinational cycle, instance port number '1' has a path to port number '0'}}
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

// -----

module  {
  // Multiple simple loops in one SCC
  // CHECK-NOT: firrtl.circuit "hasloops"
  firrtl.circuit "hasloops"   {

    // expected-error @+2 {{detected combinational cycle in a FIRRTL module}}
    // expected-remark @+1 {{this operation is part of the combinational cycle, module argument 0}}
    firrtl.module @hasloops(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
      %a = firrtl.wire  : !firrtl.uint<1>
      %b = firrtl.wire  : !firrtl.uint<1>
      %c = firrtl.wire  : !firrtl.uint<1>
      %d = firrtl.wire  : !firrtl.uint<1>
      %e = firrtl.wire  : !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %0 = firrtl.and %c, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
      %1 = firrtl.and %a, %d : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // expected-remark @+1 {{this operation is part of the combinational cycle}}
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
