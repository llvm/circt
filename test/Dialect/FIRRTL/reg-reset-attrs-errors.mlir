// RUN: circt-opt %s -split-input-file -verify-diagnostics

// The clock edge, reset type, and reset polarity attributes are required
// (front-end-explicit, no defaults). A register that omits a required attribute
// is rejected by the (ODS-generated) verifier.

firrtl.circuit "RegMissingClockEdge" {
firrtl.module @RegMissingClockEdge(in %clock: !firrtl.clock) {
  // expected-error @+1 {{'firrtl.reg' op requires attribute 'clockEdge'}}
  %r = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
}
}

// -----

firrtl.circuit "RegResetMissingClockEdge" {
firrtl.module @RegResetMissingClockEdge(in %clock: !firrtl.clock,
                                        in %reset: !firrtl.reset,
                                        in %in: !firrtl.uint<8>) {
  // expected-error @+1 {{'firrtl.regreset' op requires attribute 'clockEdge'}}
  %r = firrtl.regreset %clock, %reset, %in {resetPolarity = 0 : i32, resetType = 0 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
}
}

// -----

firrtl.circuit "RegResetMissingResetType" {
firrtl.module @RegResetMissingResetType(in %clock: !firrtl.clock,
                                        in %reset: !firrtl.reset,
                                        in %in: !firrtl.uint<8>) {
  // expected-error @+1 {{'firrtl.regreset' op requires attribute 'resetType'}}
  %r = firrtl.regreset %clock, %reset, %in {clockEdge = 0 : i32, resetPolarity = 0 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
}
}

// -----

firrtl.circuit "RegResetMissingResetPolarity" {
firrtl.module @RegResetMissingResetPolarity(in %clock: !firrtl.clock,
                                            in %reset: !firrtl.reset,
                                            in %in: !firrtl.uint<8>) {
  // expected-error @+1 {{'firrtl.regreset' op requires attribute 'resetPolarity'}}
  %r = firrtl.regreset %clock, %reset, %in {clockEdge = 0 : i32, resetType = 0 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
}
}

// -----

// Reset behavior attributes are not allowed on a reset-less register.
firrtl.circuit "RegWithResetType" {
firrtl.module @RegWithResetType(in %clock: !firrtl.clock) {
  // expected-error @+1 {{has reset attribute 'resetType', which is only valid on 'firrtl.regreset'}}
  %r = firrtl.reg %clock {clockEdge = 0 : i32, resetType = 1 : i32} : !firrtl.clock, !firrtl.uint<8>
}
}

// -----

firrtl.circuit "RegWithResetPolarity" {
firrtl.module @RegWithResetPolarity(in %clock: !firrtl.clock) {
  // expected-error @+1 {{has reset attribute 'resetPolarity', which is only valid on 'firrtl.regreset'}}
  %r = firrtl.reg %clock {clockEdge = 0 : i32, resetPolarity = 1 : i32} : !firrtl.clock, !firrtl.uint<8>
}
}

// -----

// A dual-edge (`edge`/`AtEdge`) clock is not valid synthesizable logic.
firrtl.circuit "RegDualEdgeClock" {
firrtl.module @RegDualEdgeClock(in %clock: !firrtl.clock) {
  // expected-error @+1 {{has 'clockEdge = edge' (dual-edge), which is not valid synthesizable logic; use 'posedge' or 'negedge'}}
  %r = firrtl.reg %clock {clockEdge = 2 : i32} : !firrtl.clock, !firrtl.uint<8>
}
}

// -----

firrtl.circuit "RegResetDualEdgeClock" {
firrtl.module @RegResetDualEdgeClock(in %clock: !firrtl.clock,
                                     in %reset: !firrtl.reset,
                                     in %in: !firrtl.uint<8>) {
  // expected-error @+1 {{has 'clockEdge = edge' (dual-edge), which is not valid synthesizable logic; use 'posedge' or 'negedge'}}
  %r = firrtl.regreset %clock, %reset, %in {clockEdge = 2 : i32, resetPolarity = 0 : i32, resetType = 0 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
}
}
