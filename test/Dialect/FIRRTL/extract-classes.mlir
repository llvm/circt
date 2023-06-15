// RUN: circt-opt -firrtl-extract-classes %s | FileCheck %s

firrtl.circuit "Top" {
  firrtl.module @Top() {}

  // CHECK-LABEL: firrtl.module @SomeProperties
  // CHECK-SAME: (in %in1: !firrtl.uint<1>, out %out3: !firrtl.uint<1>)
  // CHECK-NOT: firrtl.propassign
  firrtl.module @SomeProperties(
      in %in0: !firrtl.string,
      in %in1: !firrtl.uint<1>,
      out %out0: !firrtl.string,
      out %out1: !firrtl.string,
      out %out2: !firrtl.string,
      out %out3: !firrtl.uint<1>) {
    %0 = firrtl.string "hello"
    firrtl.propassign %out0, %0 : !firrtl.string
    firrtl.propassign %out1, %0 : !firrtl.string
    firrtl.propassign %out2, %in0 : !firrtl.string
    firrtl.connect %out3, %in1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// CHECK-LABEL: om.class @SomeProperties
// CHECK-SAME: (%[[P0:.+]]: !firrtl.string)
// CHECK: %[[S0:.+]] = firrtl.string "hello"
// CHECK: om.class.field @out0, %[[S0]] : !firrtl.string
// CHECK: om.class.field @out1, %[[S0]] : !firrtl.string
// CHECK: om.class.field @out2, %[[P0]] : !firrtl.string
