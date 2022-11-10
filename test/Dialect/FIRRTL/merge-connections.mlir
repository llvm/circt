// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(merge-connections)))' %s | FileCheck %s --check-prefixes=CHECK,COMMON
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(merge-connections{aggressive-merging=true})))' %s | FileCheck %s --check-prefixes=AGGRESSIVE,COMMON

firrtl.circuit "Test"   {
  // circuit Test :
  //   module Test :
  //     input a : {c: {clock: Clock, valid:UInt<1>}[2]}
  //     output b : {c: {clock: Clock, valid:UInt<1>}[2]}
  //     b <= a
  // COMMON-LABEL: firrtl.module @Test(
  // COMMON-NEXT:    firrtl.strictconnect %b, %a
  // COMMON-NEXT:  }
  firrtl.module @Test(in %a: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>, out %b: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>) {
     %0 = firrtl.subindex %a[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %1 = firrtl.subindex %b[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %2 = firrtl.subfield %0(0) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.clock
     %3 = firrtl.subfield %1(0) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.clock
     firrtl.strictconnect %3, %2 : !firrtl.clock
     %4 = firrtl.subfield %0(1) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.uint<1>
     %5 = firrtl.subfield %1(1) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.uint<1>
     firrtl.strictconnect %5, %4 : !firrtl.uint<1>
     %6 = firrtl.subindex %a[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %7 = firrtl.subindex %b[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %8 = firrtl.subfield %6(0) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.clock
     %9 = firrtl.subfield %7(0) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.clock
     firrtl.strictconnect %9, %8 : !firrtl.clock
     %10 = firrtl.subfield %6(1) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.uint<1>
     %11 = firrtl.subfield %7(1) : (!firrtl.bundle<clock: clock, valid: uint<1>>) -> !firrtl.uint<1>
     firrtl.strictconnect %11, %10 : !firrtl.uint<1>
  }

  // circuit Bar :
  //   module Bar :
  //     output a : {b: UInt<1>, c:UInt<1>}
  //     a.b <= UInt<1>(0)
  //     a.c <= UInt<1>(1)
  // COMMON-LABEL: firrtl.module @Constant(
  // COMMON-NEXT:    %c1_ui2 = firrtl.constant 1
  // COMMON-NEXT:    %0 = firrtl.bitcast %c1_ui2
  // COMMON-NEXT:    firrtl.strictconnect %a, %0
  // COMMON-NEXT:  }
  firrtl.module @Constant(out %a: !firrtl.bundle<b: uint<1>, c: uint<1>>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.subfield %a(0) : (!firrtl.bundle<b: uint<1>, c: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %a(1) : (!firrtl.bundle<b: uint<1>, c: uint<1>>) -> !firrtl.uint<1>
    firrtl.strictconnect %0, %c0_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %1, %c1_ui1 : !firrtl.uint<1>
  }

  // AGGRESSIVE-LABEL:  firrtl.module @ConcatToVector(
  // AGGRESSIVE-NEXT:     %0 = firrtl.cat %s2, %s1
  // AGGRESSIVE-NEXT:     %1 = firrtl.bitcast %0
  // AGGRESSIVE-NEXT:     firrtl.strictconnect %sink, %1
  // AGGRESSIVE-NEXT:   }
  // CHECK-LABEL:       firrtl.module @ConcatToVector(
  // CHECK-NEXT:          %0 = firrtl.subindex %sink[1]
  // CHECK-NEXT:          %1 = firrtl.subindex %sink[0]
  // CHECK-NEXT:          firrtl.strictconnect %1, %s1
  // CHECK-NEXT:          firrtl.strictconnect %0, %s2
  // CHECK-NEXT:        }

  firrtl.module @ConcatToVector(in %s1: !firrtl.uint<1>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %sink[1] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %sink[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %1, %s1 : !firrtl.uint<1>
    firrtl.strictconnect %0, %s2 : !firrtl.uint<1>
  }

  // Check that we don't use %s1 as a source value.
  // AGGRESSIVE-LABEL:   firrtl.module @FailedToUseAggregate(
  // AGGRESSIVE-NEXT:    %0 = firrtl.subindex %s1[0]
  // AGGRESSIVE-NEXT:    %1 = firrtl.cat %s2, %0
  // AGGRESSIVE-NEXT:    %2 = firrtl.bitcast %1
  // AGGRESSIVE-NEXT:    firrtl.strictconnect %sink, %2
  // AGGRESSIVE-NEXT:   }
  // CHECK-LABEL:       firrtl.module @FailedToUseAggregate(
  // CHECK-NEXT:         %0 = firrtl.subindex %sink[1]
  // CHECK-NEXT:         %1 = firrtl.subindex %s1[0]
  // CHECK-NEXT:         %2 = firrtl.subindex %sink[0]
  // CHECK-NEXT:         firrtl.strictconnect %2, %1
  // CHECK-NEXT:         firrtl.strictconnect %0, %s2
  // CHECK-NEXT:        }
  firrtl.module @FailedToUseAggregate(in %s1: !firrtl.vector<uint<1>, 2>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %sink[1] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %s1[0] : !firrtl.vector<uint<1>, 2>
    %2 = firrtl.subindex %sink[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %2, %1 : !firrtl.uint<1>
    firrtl.strictconnect %0, %s2 : !firrtl.uint<1>
  }

  // COMMON-LABEL:  firrtl.module @CheckConstant(out %c: !firrtl.vector<vector<uint<1>, 2>, 2>) {
  // COMMON-NEXT:     %c0_ui4 = firrtl.constant 0
  // COMMON-NEXT:     %0 = firrtl.bitcast %c0_ui4
  // COMMON-NEXT:     firrtl.strictconnect %c, %0
  // COMMON-NEXT:   }
  firrtl.module @CheckConstant(out %c: !firrtl.vector<vector<uint<1>, 2>, 2>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    %0 = firrtl.subindex %c[1] : !firrtl.vector<vector<uint<1>, 2>, 2>
    %1 = firrtl.subindex %0[1] : !firrtl.vector<uint<1>, 2>
    %2 = firrtl.subindex %0[0] : !firrtl.vector<uint<1>, 2>
    %3 = firrtl.subindex %c[0] : !firrtl.vector<vector<uint<1>, 2>, 2>
    %4 = firrtl.subindex %3[1] : !firrtl.vector<uint<1>, 2>
    %5 = firrtl.subindex %3[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %5, %invalid_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %4, %invalid_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %2, %invalid_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %1, %invalid_ui1 : !firrtl.uint<1>
  }
}
