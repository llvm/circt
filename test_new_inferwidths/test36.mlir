firrtl.circuit "InferComplexBundles" {
  firrtl.module @InferComplexBundles() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: bundle<v: vector<uint<3>, 10>>, b: bundle<v: vector<uint<3>, 10>>>
    %w = firrtl.wire : !firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>
    %w_a_v = firrtl.subfield %w_a[v] : !firrtl.bundle<v : vector<uint, 10>>
    %w_b = firrtl.subfield %w[b] : !firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>
    %w_b_v = firrtl.subfield %w_b[v] : !firrtl.bundle<v : vector<uint, 10>>
    firrtl.connect %w_a_v, %w_b_v : !firrtl.vector<uint, 10>, !firrtl.vector<uint, 10>
    %w_b_v_2 = firrtl.subindex %w_b_v[2] : !firrtl.vector<uint, 10>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w_b_v_2, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }
}