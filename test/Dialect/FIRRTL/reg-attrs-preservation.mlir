// Transforms that rebuild registers must preserve the clock/reset attributes.
// RUN: circt-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(firrtl.circuit(firrtl-lower-types))" \
// RUN:   | FileCheck %s --check-prefix=LOWERTYPES
// RUN: circt-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(firrtl.circuit(firrtl.module(firrtl-sfc-compat)))" \
// RUN:   | FileCheck %s --check-prefix=SFCCOMPAT

// LowerTypes splits an aggregate register into per-field registers; each must
// keep the non-default clock edge.
// LOWERTYPES-LABEL: firrtl.module @LowerTypesKeepsClockEdge
// LOWERTYPES: %r_a = firrtl.reg %clk {clockEdge = 1 : i32}
// LOWERTYPES: %r_b = firrtl.reg %clk {clockEdge = 1 : i32}
firrtl.circuit "LowerTypesKeepsClockEdge" {
firrtl.module @LowerTypesKeepsClockEdge(in %clk: !firrtl.clock,
    out %o: !firrtl.bundle<a: uint<1>, b: uint<2>>) {
  %r = firrtl.reg %clk {clockEdge = 1 : i32} : !firrtl.clock, !firrtl.bundle<a: uint<1>, b: uint<2>>
  firrtl.matchingconnect %o, %r : !firrtl.bundle<a: uint<1>, b: uint<2>>
}
}

// -----

// SFCCompat rewrites a register whose reset value is invalid into a reset-less
// register; the non-default clock edge must survive the rewrite.
// SFCCOMPAT-LABEL: firrtl.module @SFCCompatKeepsClockEdge
// SFCCOMPAT: %r = firrtl.reg %clk {clockEdge = 1 : i32}
// SFCCOMPAT-NOT: firrtl.regreset
firrtl.circuit "SFCCompatKeepsClockEdge" {
firrtl.module @SFCCompatKeepsClockEdge(in %clk: !firrtl.clock, in %rst: !firrtl.reset) {
  %inv = firrtl.invalidvalue : !firrtl.uint<8>
  %r = firrtl.regreset %clk, %rst, %inv {clockEdge = 1 : i32, resetPolarity = 0 : i32, resetType = 0 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
}
}
