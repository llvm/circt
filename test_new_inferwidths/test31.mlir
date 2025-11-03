firrtl.circuit "InferVectorSubindex" {
  firrtl.module @InferVectorSubindex(in %in : !firrtl.uint<4>, in %clk : !firrtl.clock) {
    // CHECK: firrtl.wire : !firrtl.vector<uint<4>, 10>
    // CHECK: firrtl.reg %clk : !firrtl.clock, !firrtl.vector<uint<4>, 10>
    %w = firrtl.wire : !firrtl.vector<uint, 10>
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.vector<uint, 10>
    %w_5 = firrtl.subindex %w[5] : !firrtl.vector<uint, 10>
    %r_5 = firrtl.subindex %r[5] : !firrtl.vector<uint, 10>
    firrtl.connect %w_5, %in : !firrtl.uint, !firrtl.uint<4>
    firrtl.connect %r_5, %in : !firrtl.uint, !firrtl.uint<4>
  }
}