// RUN: circt-opt %s --firrtl-lower-types --lower-firrtl-to-hw --canonicalize | FileCheck %s
firrtl.circuit "BitcastIsANop1" {

// Bitcasting bundle -> bits -> bundle should be a nop and should not shuffle
// fields. See https://github.com/llvm/circt/issues/6360
// CHECK-LABEL: hw.module @BitcastIsANop1(
firrtl.module @BitcastIsANop1(
  in %a: !firrtl.bundle<data: uint<3>, strb: uint<2>, last: uint<1>>,
  out %b: !firrtl.bundle<data: uint<3>, strb: uint<2>, last: uint<1>>
) {
  // CHECK: hw.output %a_data, %a_strb, %a_last : i3, i2, i1
  %0 = firrtl.bitcast %a : (!firrtl.bundle<data: uint<3>, strb: uint<2>, last: uint<1>>) -> !firrtl.uint<6>
  %1 = firrtl.bitcast %0 : (!firrtl.uint<6>) -> !firrtl.bundle<data: uint<3>, strb: uint<2>, last: uint<1>>
  firrtl.strictconnect %b, %1 : !firrtl.bundle<data: uint<3>, strb: uint<2>, last: uint<1>>
}

// Bitcasting bits -> bundle -> bits should be a nop and should not shuffle
// fields. See https://github.com/llvm/circt/issues/6360
// CHECK-LABEL: hw.module @BitcastIsANop2(
firrtl.module @BitcastIsANop2(
  in %a: !firrtl.uint<6>,
  out %b: !firrtl.uint<6>
) {
  // CHECK: hw.output %a : i6
  %0 = firrtl.bitcast %a : (!firrtl.uint<6>) -> !firrtl.bundle<data: uint<3>, strb: uint<2>, last: uint<1>>
  %1 = firrtl.bitcast %0 : (!firrtl.bundle<data: uint<3>, strb: uint<2>, last: uint<1>>) -> !firrtl.uint<6>
  firrtl.strictconnect %b, %1 : !firrtl.uint<6>
}

}
