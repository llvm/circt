firrtl.circuit "InferVectorSubaccess" {
  firrtl.module @InferVectorSubaccess(in %in : !firrtl.uint<4>, in %addr : !firrtl.uint<32>, in %clk : !firrtl.clock) {
    // CHECK: firrtl.wire : !firrtl.vector<uint<4>, 10>
    // CHECK: firrtl.reg %clk : !firrtl.clock, !firrtl.vector<uint<4>, 10>
    %w = firrtl.wire : !firrtl.vector<uint, 10>
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.vector<uint, 10>
    %w_addr = firrtl.subaccess %w[%addr] : !firrtl.vector<uint, 10>, !firrtl.uint<32>
    %r_addr = firrtl.subaccess %r[%addr] : !firrtl.vector<uint, 10>, !firrtl.uint<32>
    firrtl.connect %w_addr, %in : !firrtl.uint, !firrtl.uint<4>
    firrtl.connect %r_addr, %in : !firrtl.uint, !firrtl.uint<4>
  }
}