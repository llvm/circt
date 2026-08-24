// REQUIRES: libz3
// REQUIRES: circt-bmc-jit
// RUN: circt-bmc %s -b 3 --module FormalTop --shared-libs=%libz3 | FileCheck %s --check-prefix=MULTI
// RUN: circt-bmc %s -b 3 --module FormalTop --shared-libs=%libz3 --print-only-first-counterexample | FileCheck %s --check-prefix=ONE

// MULTI: counterexample for FormalTop:
// MULTI: counterexample for FormalTop:

// ONE: counterexample for FormalTop:
// ONE-NEXT: cycle 0:
// ONE-NEXT:   count = 0x2
// ONE-NEXT: Assertion can be violated!
// ONE-NOT: counterexample for FormalTop:

hw.module @FormalTop(in %count : i2) {
  %two = hw.constant 2 : i2
  %notTwo = comb.icmp ne %count, %two : i2
  verif.assert %notTwo : i1
  hw.output
}
