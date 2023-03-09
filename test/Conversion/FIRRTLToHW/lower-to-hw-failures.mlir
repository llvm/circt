
  // CHECK-LABEL: hw.module private @connectNarrowUIntVector
  firrtl.module private @connectNarrowUIntVector(in %clock: !firrtl.clock, in %a: !firrtl.vector<uint<1>, 1>, out %b: !firrtl.vector<uint<3>, 1>) {
    %r1 = firrtl.reg %clock  : !firrtl.vector<uint<2>, 1>
    firrtl.connect %r1, %a : !firrtl.vector<uint<2>, 1>, !firrtl.vector<uint<1>, 1>
    firrtl.connect %b, %r1 : !firrtl.vector<uint<3>, 1>, !firrtl.vector<uint<2>, 1>
    // CHECK:      %r1 = seq.firreg %3 clock %clock : !hw.array<1xi2>
    // CHECK-NEXT: %1 = hw.array_get %a[%false] : !hw.array<1xi1>
    // CHECK-NEXT: %2 = comb.concat %false, %1 : i1, i1
    // CHECK-NEXT: %3 = hw.array_create %2 : i2
    // CHECK-NEXT: %4 = hw.array_get %r1[%false] : !hw.array<1xi2>
    // CHECK-NEXT: %5 = comb.concat %false, %4 : i1, i2
    // CHECK-NEXT: %6 = hw.array_create %5 : i3
    // CHECK-NEXT: sv.assign %.b.output, %6 : !hw.array<1xi3>
    // CHECK-NEXT: hw.output %0 : !hw.array<1xi3>
  }

  // CHECK-LABEL: hw.module private @connectNarrowSIntVector
  firrtl.module private @connectNarrowSIntVector(in %clock: !firrtl.clock, in %a: !firrtl.vector<sint<1>, 1>, out %b: !firrtl.vector<sint<3>, 1>) {
    %r1 = firrtl.reg %clock  : !firrtl.vector<sint<2>, 1>
    firrtl.connect %r1, %a : !firrtl.vector<sint<2>, 1>, !firrtl.vector<sint<1>, 1>
    firrtl.connect %b, %r1 : !firrtl.vector<sint<3>, 1>, !firrtl.vector<sint<2>, 1>
    // CHECK:      %r1 = seq.firreg %3 clock %clock : !hw.array<1xi2>
    // CHECK-NEXT: %1 = hw.array_get %a[%false] : !hw.array<1xi1>
    // CHECK-NEXT: %2 = comb.concat %1, %1 : i1, i1
    // CHECK-NEXT: %3 = hw.array_create %2 : i2
    // CHECK-NEXT: %4 = hw.array_get %r1[%false] : !hw.array<1xi2>
    // CHECK-NEXT: %5 = comb.extract %4 from 1 : (i2) -> i1
    // CHECK-NEXT: %6 = comb.concat %5, %4 : i1, i2
    // CHECK-NEXT: %7 = hw.array_create %6 : i3
    // CHECK-NEXT: sv.assign %.b.output, %7 : !hw.array<1xi3>
    // CHECK-NEXT: hw.output %0 : !hw.array<1xi3>
  }