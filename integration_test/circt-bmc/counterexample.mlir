// REQUIRES: libz3
// REQUIRES: circt-bmc-jit
// RUN: circt-bmc %s -b 3 --module FormalTop --shared-libs=%libz3 | FileCheck %s

// CHECK: counterexample for FormalTop:
// CHECK-NEXT: cycle 0:
// CHECK-NEXT:   count = 0x2
// CHECK-NEXT: Assertion can be violated!
// CHECK-NOT: counterexample for FormalTop:

hw.module @FormalTop(in %count : i2) {
  %two = hw.constant 2 : i2
  %notTwo = comb.icmp ne %count, %two : i2
  verif.assert %notTwo : i1
  hw.output
}
