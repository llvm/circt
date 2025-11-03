firrtl.circuit "InferNode2" {
  firrtl.module @InferNode2() {
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %w = firrtl.wire : !firrtl.uint
    firrtl.connect %w, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>

    %node2 = firrtl.node %w : !firrtl.uint

    %w1 = firrtl.wire : !firrtl.uint
    firrtl.connect %w1, %node2 : !firrtl.uint, !firrtl.uint
  }
}