// RUN: circt-opt -firrtl-extract-classes %s | FileCheck %s

firrtl.circuit "Top" {
  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top() {
    // CHECK-NOT: firrtl.instance all
    %all_in0, %all_out0 = firrtl.instance all @AllProperties(
      in in0: !firrtl.string,
      out out0: !firrtl.string)

    // CHECK: %some_in1, %some_out3 = firrtl.instance some
    %some_in0, %some_in1, %some_out0, %some_out1, %some_out2, %some_out3 = firrtl.instance some @SomeProperties(
      in in0: !firrtl.string,
      in in1: !firrtl.uint<1>,
      out out0: !firrtl.string,
      out out1: !firrtl.string,
      out out2: !firrtl.string,
      out out3: !firrtl.uint<1>)

    // CHECK: %no_in0, %no_out0 = firrtl.instance no
    %no_in0, %no_out0 = firrtl.instance no @NoProperties(
      in in0: !firrtl.uint<1>,
      out out0: !firrtl.uint<1>)

    // CHECK-NOT: firrtl.propassign
    firrtl.propassign %some_in0, %all_out0 : !firrtl.string
  }

  // CHECK-NOT: @AllProperties
  firrtl.module @AllProperties(
      in %in0: !firrtl.string,
      out %out0: !firrtl.string) {
    firrtl.propassign %out0, %in0 : !firrtl.string
  }

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

  // CHECK-LABEL: firrtl.module @NoProperties
  // CHECK-SAME: (in %in0: !firrtl.uint<1>, out %out0: !firrtl.uint<1>)
  // CHECK: firrtl.connect
  firrtl.module @NoProperties(
      in %in0: !firrtl.uint<1>,
      out %out0: !firrtl.uint<1>) {
    firrtl.connect %out0, %in0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// CHECK-LABEL: om.class @AllProperties
// CHECK-SAME: (%[[P0:.+]]: !firrtl.string)
// CHECK: om.class.field @out0, %[[P0]] : !firrtl.string

// CHECK-LABEL: om.class @SomeProperties
// CHECK-SAME: (%[[P0:.+]]: !firrtl.string)
// CHECK: %[[S0:.+]] = firrtl.string "hello"
// CHECK: om.class.field @out0, %[[S0]] : !firrtl.string
// CHECK: om.class.field @out1, %[[S0]] : !firrtl.string
// CHECK: om.class.field @out2, %[[P0]] : !firrtl.string
