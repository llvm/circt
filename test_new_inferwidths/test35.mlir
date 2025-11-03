firrtl.circuit "InferElementAfterVector" {
  firrtl.module @InferElementAfterVector() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: vector<uint<10>, 10>, b: uint<3>>
    %w = firrtl.wire : !firrtl.bundle<a: vector<uint<10>, 10>, b :uint>
    %w_a = firrtl.subfield %w[b] : !firrtl.bundle<a: vector<uint<10>, 10>, b: uint>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w_a, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }
}