// RUN: circt-opt %s --verify-roundtrip | FileCheck %s

// Round-trip test for the `initial` time-zero simulation value attribute on
// `firrtl.reg` and `firrtl.regreset`.

firrtl.circuit "RegInitial" {
  // CHECK-LABEL: firrtl.module @RegInitial
  firrtl.module @RegInitial(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                            in %d: !firrtl.uint<8>, out %q: !firrtl.uint<8>) {
    // CHECK: %r = firrtl.reg %clock {initial = 5 : ui8} : !firrtl.clock, !firrtl.uint<8>
    %r = firrtl.reg %clock {initial = 5 : ui8} : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>

    %c7 = firrtl.constant 7 : !firrtl.uint<8>
    // CHECK: %s = firrtl.regreset %clock, %reset, %c7_ui8 {initial = 0 : ui8}
    %s = firrtl.regreset %clock, %reset, %c7 {initial = 0 : ui8} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>

    // Signed register with signed initial value.
    // CHECK: %t = firrtl.reg %clock {initial = -1 : si8} : !firrtl.clock, !firrtl.sint<8>
    %t = firrtl.reg %clock {initial = -1 : si8} : !firrtl.clock, !firrtl.sint<8>

    firrtl.matchingconnect %q, %r : !firrtl.uint<8>
  }
}
