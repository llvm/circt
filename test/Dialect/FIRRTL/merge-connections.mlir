// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-merge-connections)))' %s | FileCheck %s --check-prefixes=CHECK,COMMON
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-merge-connections{aggressive-merging=true})))' %s | FileCheck %s --check-prefixes=AGGRESSIVE,COMMON

firrtl.circuit "Test"   {
  firrtl.layer @A bind {}
  // circuit Test :
  //   module Test :
  //     input a : {c: {clock: Clock, valid:UInt<1>}[2]}
  //     output b : {c: {clock: Clock, valid:UInt<1>}[2]}
  //     b <= a
  // COMMON-LABEL: firrtl.module @Test(
  // COMMON-NEXT:    firrtl.matchingconnect %b, %a
  // COMMON-NEXT:  }
  firrtl.module @Test(in %a: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>, out %b: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>) {
     %0 = firrtl.subindex %a[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %1 = firrtl.subindex %b[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %2 = firrtl.subfield %0[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %3 = firrtl.subfield %1[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     firrtl.matchingconnect %3, %2 : !firrtl.clock
     %4 = firrtl.subfield %0[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %5 = firrtl.subfield %1[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     firrtl.matchingconnect %5, %4 : !firrtl.uint<1>
     %6 = firrtl.subindex %a[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %7 = firrtl.subindex %b[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %8 = firrtl.subfield %6[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %9 = firrtl.subfield %7[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     firrtl.matchingconnect %9, %8 : !firrtl.clock
     %10 = firrtl.subfield %6[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %11 = firrtl.subfield %7[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     firrtl.matchingconnect %11, %10 : !firrtl.uint<1>
  }

  // This is the same as @Test except it puts the destination into a layer.
  //
  // COMMON-LABEL: firrtl.module @Layers(
  // COMMON-NEXT:    firrtl.layerblock @A {
  // COMMON-NEXT:      %b = firrtl.wire
  // COMMON-NEXT:      firrtl.matchingconnect %b, %a
  // COMMON-NEXT:    }
  // COMMON-NEXT:  }
  firrtl.module @Layers(in %a: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>) {
    firrtl.layerblock @A {
      %b = firrtl.wire : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
      %0 = firrtl.subindex %a[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
      %1 = firrtl.subindex %b[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
      %2 = firrtl.subfield %0[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
      %3 = firrtl.subfield %1[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
      firrtl.matchingconnect %3, %2 : !firrtl.clock
      %4 = firrtl.subfield %0[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
      %5 = firrtl.subfield %1[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
      firrtl.matchingconnect %5, %4 : !firrtl.uint<1>
      %6 = firrtl.subindex %a[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
      %7 = firrtl.subindex %b[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
      %8 = firrtl.subfield %6[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
      %9 = firrtl.subfield %7[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
      firrtl.matchingconnect %9, %8 : !firrtl.clock
      %10 = firrtl.subfield %6[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
      %11 = firrtl.subfield %7[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
      firrtl.matchingconnect %11, %10 : !firrtl.uint<1>
    }
  }

  // circuit Bar :
  //   module Bar :
  //     output a : {b: UInt<1>, c:UInt<1>}
  //     a.b <= UInt<1>(0)
  //     a.c <= UInt<1>(1)
  // COMMON-LABEL: firrtl.module @Constant(
  // COMMON-NEXT:    %0 = firrtl.aggregateconstant [0 : ui1, 1 : ui1]
  // COMMON-NEXT:    firrtl.matchingconnect %a, %0
  // COMMON-NEXT:  }
  firrtl.module @Constant(out %a: !firrtl.bundle<b: uint<1>, c: uint<1>>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.subfield %a[b] : !firrtl.bundle<b: uint<1>, c: uint<1>>
    %1 = firrtl.subfield %a[c] : !firrtl.bundle<b: uint<1>, c: uint<1>>
    firrtl.matchingconnect %0, %c0_ui1 : !firrtl.uint<1>
    firrtl.matchingconnect %1, %c1_ui1 : !firrtl.uint<1>
  }

  // AGGRESSIVE-LABEL:  firrtl.module @ConcatToVector(
  // AGGRESSIVE-NEXT:     %0 = firrtl.vectorcreate %s1, %s2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  // AGGRESSIVE-NEXT:     firrtl.matchingconnect %sink, %0
  // AGGRESSIVE-NEXT:   }
  // CHECK-LABEL:       firrtl.module @ConcatToVector(
  // CHECK-NEXT:          %0 = firrtl.subindex %sink[1]
  // CHECK-NEXT:          %1 = firrtl.subindex %sink[0]
  // CHECK-NEXT:          firrtl.matchingconnect %1, %s1
  // CHECK-NEXT:          firrtl.matchingconnect %0, %s2
  // CHECK-NEXT:        }

  firrtl.module @ConcatToVector(in %s1: !firrtl.uint<1>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %sink[1] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %sink[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %1, %s1 : !firrtl.uint<1>
    firrtl.matchingconnect %0, %s2 : !firrtl.uint<1>
  }

  // Check that we don't use %s1 as a source value.
  // AGGRESSIVE-LABEL:   firrtl.module @FailedToUseAggregate(
  // AGGRESSIVE-NEXT:    %0 = firrtl.subindex %s1[0]
  // AGGRESSIVE-NEXT:    %1 = firrtl.vectorcreate %0, %s2
  // AGGRESSIVE-NEXT:    firrtl.matchingconnect %sink, %1
  // AGGRESSIVE-NEXT:   }
  // CHECK-LABEL:       firrtl.module @FailedToUseAggregate(
  // CHECK-NEXT:         %0 = firrtl.subindex %sink[1]
  // CHECK-NEXT:         %1 = firrtl.subindex %s1[0]
  // CHECK-NEXT:         %2 = firrtl.subindex %sink[0]
  // CHECK-NEXT:         firrtl.matchingconnect %2, %1
  // CHECK-NEXT:         firrtl.matchingconnect %0, %s2
  // CHECK-NEXT:        }
  firrtl.module @FailedToUseAggregate(in %s1: !firrtl.vector<uint<1>, 2>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %sink[1] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %s1[0] : !firrtl.vector<uint<1>, 2>
    %2 = firrtl.subindex %sink[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %2, %1 : !firrtl.uint<1>
    firrtl.matchingconnect %0, %s2 : !firrtl.uint<1>
  }


  // No merging with non-passive type.
  // COMMON-LABEL: firrtl.module private @DUT
  // COMMON-NEXT:    %p = firrtl.wire
  // COMMON-NEXT:    %0 = firrtl.subfield
  // COMMON-NEXT:    firrtl.matchingconnect %0, %x_a
  // COMMON-NEXT:    %1 = firrtl.subfield
  // COMMON-NEXT:    firrtl.matchingconnect %x_b, %1
  // COMMON-NEXT:    firrtl.matchingconnect %y_a, %0
  // COMMON-NEXT:    firrtl.matchingconnect %1, %y_b
  // COMMON-NEXT:  }
  firrtl.module private @DUT(in %x_a: !firrtl.uint<2>,
                             out %x_b: !firrtl.uint<2>,
                             out %y_a: !firrtl.uint<2>,
                             in %y_b: !firrtl.uint<2>) {
    %p = firrtl.wire : !firrtl.bundle<a: uint<2>, b flip: uint<2>>
    %0 = firrtl.subfield %p[a] : !firrtl.bundle<a: uint<2>, b flip: uint<2>>
    firrtl.matchingconnect %0, %x_a : !firrtl.uint<2>
    %1 = firrtl.subfield %p[b] : !firrtl.bundle<a: uint<2>, b flip: uint<2>>
    firrtl.matchingconnect %x_b, %1 : !firrtl.uint<2>
    firrtl.matchingconnect %y_a, %0 : !firrtl.uint<2>
    firrtl.matchingconnect %1, %y_b : !firrtl.uint<2>
  }

  // Don't create aggregateconstant of non-passive. #6259.
  // COMMON-LABEL: @Issue6259
  // COMMON-NOT: aggregateconstant
  // COMMON: }
  firrtl.module private @Issue6259(out %a: !firrtl.rwprobe<bundle<a: uint<1>, b: uint<2>>>) {
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %w, %w_ref = firrtl.wire forceable {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.bundle<a: uint<1>, b flip: uint<2>>, !firrtl.rwprobe<bundle<a: uint<1>, b: uint<2>>>
    %0 = firrtl.subfield %w[b] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    %1 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    firrtl.matchingconnect %1, %c0_ui1 : !firrtl.uint<1>
    firrtl.matchingconnect %0, %c0_ui2 : !firrtl.uint<2>
    firrtl.ref.define %a, %w_ref : !firrtl.rwprobe<bundle<a: uint<1>, b: uint<2>>>
  }

  // COMMON-LABEL: @Alias
  firrtl.module @Alias(in %i: !firrtl.alias<MyBundle, bundle<f: uint<1>, b: uint<1>>>, out %o: !firrtl.alias<MyBundle, bundle<f: uint<1>, b: uint<1>>>) {
    // CHECK-NEXT:  firrtl.matchingconnect %o, %i
    %0 = firrtl.subfield %i[f] : !firrtl.alias<MyBundle, bundle<f: uint<1>, b: uint<1>>>
    %1 = firrtl.subfield %o[f] : !firrtl.alias<MyBundle, bundle<f: uint<1>, b: uint<1>>>
    firrtl.matchingconnect %1, %0 : !firrtl.uint<1>
    %2 = firrtl.subfield %i[b] : !firrtl.alias<MyBundle, bundle<f: uint<1>, b: uint<1>>>
    %3 = firrtl.subfield %o[b] : !firrtl.alias<MyBundle, bundle<f: uint<1>, b: uint<1>>>
    firrtl.matchingconnect %3, %2 : !firrtl.uint<1>
  }
}
