// Round-trip and lowering tests for explicit FIRRTL register clock edges.
// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --pass-pipeline="builtin.module(lower-firrtl-to-hw)" \
// RUN:   | FileCheck %s --check-prefix=LOWER

firrtl.circuit "RegAttrs" {
// CHECK-LABEL: firrtl.module @RegAttrs
firrtl.module @RegAttrs(in %clock: !firrtl.clock,
                        in %sreset: !firrtl.uint<1>,
                        in %in: !firrtl.uint<8>) {
  // LOWER-LABEL: hw.module @RegAttrs
  // A plain register with default clock edge elides the attribute.
  // CHECK: %r0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  %r0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>

  // A plain register with a non-default (negedge) clock edge round-trips.
  // CHECK: %r1 = firrtl.reg negedge %clock : !firrtl.clock, !firrtl.uint<8>
  %r1 = firrtl.reg negedge %clock : !firrtl.clock, !firrtl.uint<8>

  // A regreset with a non-default clock edge preserves it too.
  // CHECK: %r2 = firrtl.regreset negedge %clock, %sreset, %in : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
  %r2 = firrtl.regreset negedge %clock, %sreset, %in : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>

  // A regreset with the default clock edge elides the attribute.
  // CHECK: %r3 = firrtl.regreset %clock, %sreset, %in : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
  %r3 = firrtl.regreset %clock, %sreset, %in : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>

  // A dual-edge register round-trips.
  // CHECK: %r4 = firrtl.reg edge %clock : !firrtl.clock, !firrtl.uint<8>
  %r4 = firrtl.reg edge %clock : !firrtl.clock, !firrtl.uint<8>

  // LOWER-COUNT-2: {clockEdge = 1 : i32}
  // LOWER: {clockEdge = 2 : i32}
}
}
