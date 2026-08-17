// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "WidthMismatch" {
  firrtl.module @WidthMismatch(in %clock: !firrtl.clock) {
    // expected-error @below {{'initial' value bitwidth (16) doesn't match register type width (8)}}
    %r = firrtl.reg %clock {initial = 5 : ui16} : !firrtl.clock, !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "SignMismatch" {
  firrtl.module @SignMismatch(in %clock: !firrtl.clock) {
    // expected-error @below {{'initial' value has wrong sign}}
    %r = firrtl.reg %clock {initial = 5 : si8} : !firrtl.clock, !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "AggregateInitial" {
  firrtl.module @AggregateInitial(in %clock: !firrtl.clock) {
    // expected-error @below {{'initial' value is only supported on ground-type registers}}
    %r = firrtl.reg %clock {initial = 5 : ui8} : !firrtl.clock, !firrtl.vector<uint<8>, 2>
  }
}

// -----

firrtl.circuit "RegResetWidthMismatch" {
  firrtl.module @RegResetWidthMismatch(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    // expected-error @below {{'initial' value bitwidth (16) doesn't match register type width (8)}}
    %r = firrtl.regreset %clock, %reset, %c0 {initial = 5 : ui16} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
  }
}
