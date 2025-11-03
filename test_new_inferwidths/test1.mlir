firrtl.circuit "InferNode" {
  firrtl.module @InferNode() {
    %w = firrtl.wire : !firrtl.uint
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    // CHECK: %node = firrtl.node %w : !firrtl.uint<3>
    %node = firrtl.node %c2_ui3 : !firrtl.uint<3>
  }
}