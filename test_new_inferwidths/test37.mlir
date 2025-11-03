firrtl.circuit "InferComplexVectors" {
  firrtl.module @InferComplexVectors() {
    // CHECK: %w = firrtl.wire : !firrtl.vector<bundle<a: uint<3>, b: uint<3>>, 10>
    %w = firrtl.wire : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_2 = firrtl.subindex %w[2] : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_2_a = firrtl.subfield %w_2[a] : !firrtl.bundle<a: uint, b: uint>
    %w_4 = firrtl.subindex %w[4] : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_4_b = firrtl.subfield %w_4[b] : !firrtl.bundle<a: uint, b: uint>
    firrtl.connect %w_4_b, %w_2_a : !firrtl.uint, !firrtl.uint
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w_2_a, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }
}