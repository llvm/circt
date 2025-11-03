firrtl.circuit "InferInvalidValue" {
  firrtl.module @InferInvalidValue(out %out: !firrtl.uint) {
    // CHECK: %invalid_ui6 = firrtl.invalidvalue : !firrtl.uint<6>
    %invalid_ui = firrtl.invalidvalue : !firrtl.uint
    %c42_ui = firrtl.constant 42 : !firrtl.uint
    firrtl.connect %out, %invalid_ui : !firrtl.uint, !firrtl.uint
    firrtl.connect %out, %c42_ui : !firrtl.uint, !firrtl.uint

    // Check that invalid values are inferred to width zero if not used in a
    // connect.
    // CHECK: firrtl.invalidvalue : !firrtl.uint<0>
    // CHECK: firrtl.invalidvalue : !firrtl.bundle<x: uint<0>>
    // CHECK: firrtl.invalidvalue : !firrtl.vector<uint<0>, 2>
    %invalid_0 = firrtl.invalidvalue : !firrtl.uint
    %invalid_1 = firrtl.invalidvalue : !firrtl.bundle<x: uint>
    %invalid_2 = firrtl.invalidvalue : !firrtl.vector<uint, 2>
  }
}