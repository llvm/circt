firrtl.circuit "InferEmptyBundle" {
  firrtl.module @InferEmptyBundle(in %in : !firrtl.uint<3>) {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: bundle<>, b: uint<3>>
    %w = firrtl.wire : !firrtl.bundle<a: bundle<>, b: uint>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: bundle<>, b: uint>
    %w_b = firrtl.subfield %w[b] : !firrtl.bundle<a: bundle<>, b: uint>
    firrtl.connect %w_b, %in : !firrtl.uint, !firrtl.uint<3>
  }
}