// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-full-reset))' --split-input-file %s | FileCheck %s

// Basic async full-reset: reset-less register becomes regreset.
// CHECK-LABEL: firrtl.module @AsyncFullReset
firrtl.circuit "AsyncFullReset" {
  firrtl.module @AsyncFullReset(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset
          [{class = "circt.FullResetAnnotation", resetType = "async"}],
      in %in: !firrtl.uint<8>) {
    // CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %reg, %in : !firrtl.uint<8>
  }
}

// -----
// Exclude annotation is consumed; registers stay reset-less.
// CHECK-LABEL: firrtl.module @Excluded
// CHECK-NOT: ExcludeFromFullResetAnnotation
// CHECK: %reg = firrtl.reg %clock
firrtl.circuit "Excluded" {
  firrtl.module @Excluded(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>)
      attributes {annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]} {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %reg, %in : !firrtl.uint<8>
  }
}

// -----
// Child inherits async domain; reset is wired through an added port.
// CHECK-LABEL: firrtl.module @Child
// CHECK-SAME: in %reset: !firrtl.asyncreset
// CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
// CHECK-LABEL: firrtl.module @Nested
// CHECK: firrtl.matchingconnect %child_reset, %reset
firrtl.circuit "Nested" {
  firrtl.module @Child(in %clock: !firrtl.clock) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  firrtl.module @Nested(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset
          [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    %child_clock = firrtl.instance child @Child(in clock: !firrtl.clock)
    firrtl.matchingconnect %child_clock, %clock : !firrtl.clock
  }
}
