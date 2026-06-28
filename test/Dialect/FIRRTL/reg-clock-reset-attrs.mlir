// Round-trip test for the explicit clock/reset attributes on FIRRTL registers.
// RUN: circt-opt %s | circt-opt | FileCheck %s

firrtl.circuit "RegAttrs" {
// CHECK-LABEL: firrtl.module @RegAttrs
firrtl.module @RegAttrs(in %clock: !firrtl.clock,
                        in %areset: !firrtl.reset,
                        in %sreset: !firrtl.reset,
                        in %in: !firrtl.uint<8>) {
  // A plain register with default (posedge) clock edge prints the explicit attribute.
  // CHECK: %r0 = firrtl.reg %clock {clockEdge = 0 : i32} : !firrtl.clock, !firrtl.uint<8>
  %r0 = firrtl.reg %clock {clockEdge = 0 : i32} : !firrtl.clock, !firrtl.uint<8>

  // A plain register with a non-default (negedge) clock edge round-trips.
  // CHECK: %r1 = firrtl.reg %clock {clockEdge = 1 : i32} : !firrtl.clock, !firrtl.uint<8>
  %r1 = firrtl.reg %clock {clockEdge = 1 : i32} : !firrtl.clock, !firrtl.uint<8>

  // A regreset with explicit negedge clock, async reset kind, and active-low
  // (NegReset) polarity round-trips with all three attributes preserved.
  // CHECK: %r2 = firrtl.regreset %clock, %areset, %in {clockEdge = 1 : i32, resetPolarity = 1 : i32, resetType = 1 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
  %r2 = firrtl.regreset %clock, %areset, %in {clockEdge = 1 : i32, resetType = 1 : i32, resetPolarity = 1 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>

  // A regreset with all defaults prints all three explicit attributes.
  // CHECK: %r3 = firrtl.regreset %clock, %sreset, %in {clockEdge = 0 : i32, resetPolarity = 0 : i32, resetType = 0 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
  %r3 = firrtl.regreset %clock, %sreset, %in {clockEdge = 0 : i32, resetPolarity = 0 : i32, resetType = 0 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
}
}
