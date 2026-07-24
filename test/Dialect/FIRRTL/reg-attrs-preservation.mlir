// Transforms that rebuild registers must preserve the clock-edge attribute.
// RUN: circt-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(firrtl.circuit(firrtl-lower-types))" \
// RUN:   | FileCheck %s --check-prefix=LOWERTYPES
// RUN: circt-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(firrtl.circuit(firrtl.module(firrtl-sfc-compat)))" \
// RUN:   | FileCheck %s --check-prefix=SFCCOMPAT
// RUN: circt-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(firrtl.circuit(firrtl-infer-resets))" \
// RUN:   | FileCheck %s --check-prefix=INFERRESETS

// LowerTypes splits an aggregate register into per-field registers; each must
// keep the non-default clock edge.
// LOWERTYPES-LABEL: firrtl.module @LowerTypesKeepsClockEdge
// LOWERTYPES: %r_a = firrtl.reg negedge %clk
// LOWERTYPES: %r_b = firrtl.reg negedge %clk
firrtl.circuit "LowerTypesKeepsClockEdge" {
firrtl.module @LowerTypesKeepsClockEdge(in %clk: !firrtl.clock,
    out %o: !firrtl.bundle<a: uint<1>, b: uint<2>>) {
  %r = firrtl.reg negedge %clk : !firrtl.clock, !firrtl.bundle<a: uint<1>, b: uint<2>>
  firrtl.matchingconnect %o, %r : !firrtl.bundle<a: uint<1>, b: uint<2>>
}
}

// -----

// SFCCompat rewrites a register whose reset value is invalid into a reset-less
// register; the non-default clock edge must survive the rewrite.
// SFCCOMPAT-LABEL: firrtl.module @SFCCompatKeepsClockEdge
// SFCCOMPAT: %r = firrtl.reg negedge %clk
// SFCCOMPAT-NOT: firrtl.regreset
firrtl.circuit "SFCCompatKeepsClockEdge" {
firrtl.module @SFCCompatKeepsClockEdge(in %clk: !firrtl.clock, in %rst: !firrtl.uint<1>) {
  %inv = firrtl.invalidvalue : !firrtl.uint<8>
  %r = firrtl.regreset negedge %clk, %rst, %inv : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
}
}

// -----

// InferResets converts a reset-less register into a regreset when a full async
// reset is added; the non-default clock edge must survive.
// INFERRESETS-LABEL: firrtl.module @InferResetsKeepsClockEdge
// INFERRESETS: firrtl.regreset negedge %clock, %reset, {{.*}}
firrtl.circuit "InferResetsKeepsClockEdge" {
firrtl.module @InferResetsKeepsClockEdge(in %clock: !firrtl.clock,
    in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}],
    in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
  %r = firrtl.reg negedge %clock : !firrtl.clock, !firrtl.uint<8>
  firrtl.matchingconnect %r, %in : !firrtl.uint<8>
  firrtl.matchingconnect %out, %r : !firrtl.uint<8>
}
}
