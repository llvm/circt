firrtl.circuit "InferBundle" {
  firrtl.module @InferBundle(in %in : !firrtl.uint<3>, in %clk : !firrtl.clock) {
    // CHECK: firrtl.wire : !firrtl.bundle<a: uint<3>>
    // CHECK: firrtl.reg %clk : !firrtl.clock, !firrtl.bundle<a: uint<3>>
    %w = firrtl.wire : !firrtl.bundle<a: uint>
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.bundle<a: uint>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint>
    %r_a = firrtl.subfield %r[a] : !firrtl.bundle<a: uint>
    firrtl.connect %w_a, %in : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %r_a, %in : !firrtl.uint, !firrtl.uint<3>
  }
}