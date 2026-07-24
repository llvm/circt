// RUN: circt-translate --export-firrtl --verify-diagnostics \
// RUN:   --split-input-file %s

firrtl.circuit "NegEdge" {
  firrtl.module @NegEdge(in %clock: !firrtl.clock) {
    // expected-error @below {{'firrtl.reg' op has a clock edge that cannot be represented in FIRRTL text}}
    %reg = firrtl.reg negedge %clock : !firrtl.clock, !firrtl.uint<1>
  }
}
