// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-sizeof)), lower-firrtl-to-hw)' --verify-diagnostics --split-input-file %s

firrtl.circuit "SizeOf" {
  // CHECK-LABEL: hw.module @SizeOf
  firrtl.module @SizeOf(
    out %size8: !firrtl.uint<32>,
    out %sizeBundle: !firrtl.uint<32>,
    out %sizeZero: !firrtl.uint<32>
  ) {
    %uint8 = firrtl.wire : !firrtl.uint<8>
    %size8Value = firrtl.int.sizeof %uint8 :
        (!firrtl.uint<8>) -> !firrtl.uint<32>
    // CHECK-DAG: hw.constant 8 : i32
    firrtl.matchingconnect %size8, %size8Value : !firrtl.uint<32>

    %bundle = firrtl.wire : !firrtl.bundle<a: uint<3>, b: uint<4>>
    %sizeBundleValue = firrtl.int.sizeof %bundle :
        (!firrtl.bundle<a: uint<3>, b: uint<4>>) -> !firrtl.uint<32>
    // CHECK-DAG: hw.constant 7 : i32
    firrtl.matchingconnect %sizeBundle, %sizeBundleValue : !firrtl.uint<32>

    %zero = firrtl.wire : !firrtl.uint<0>
    %sizeZeroValue = firrtl.int.sizeof %zero :
        (!firrtl.uint<0>) -> !firrtl.uint<32>
    // CHECK-DAG: hw.constant 0 : i32
    firrtl.matchingconnect %sizeZero, %sizeZeroValue : !firrtl.uint<32>
  }
}