// RUN: circt-opt -lower-firrtl-to-hw %s | FileCheck %s

firrtl.circuit "RegInitial" {
  // CHECK-LABEL: hw.module @RegInitial
  firrtl.module @RegInitial(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                            in %d: !firrtl.uint<8>, out %q: !firrtl.uint<8>) {
    // CHECK: seq.firreg %d clock %clock preset 5 : i8
    %r = firrtl.reg %clock {initial = 5 : ui8} : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>

    %c7 = firrtl.constant 7 : !firrtl.uint<8>
    // CHECK: seq.firreg {{.*}} clock %clock reset sync %reset, {{.*}} preset 0 : i8
    %s = firrtl.regreset %clock, %reset, %c7 {initial = 0 : ui8} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>

    firrtl.matchingconnect %q, %r : !firrtl.uint<8>
  }
}
