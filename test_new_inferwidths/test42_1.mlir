firrtl.circuit "ForeignTypes" {
  firrtl.module @ForeignTypes(in %a: !firrtl.uint<42>, out %b: !firrtl.uint) {
    %0 = firrtl.wire : index
    %1 = firrtl.wire : index
    firrtl.matchingconnect %0, %1 : index
    firrtl.connect %b, %a : !firrtl.uint, !firrtl.uint<42>
    // CHECK-NEXT: [[W0:%.+]] = firrtl.wire : index
    // CHECK-NEXT: [[W1:%.+]] = firrtl.wire : index
    // CHECK-NEXT: firrtl.matchingconnect [[W0]], [[W1]] : index
  }
}