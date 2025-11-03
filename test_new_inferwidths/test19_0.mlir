firrtl.circuit "TruncateConnect" {
  firrtl.module @TruncateConnect() {
    %w = firrtl.wire  : !firrtl.uint
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %w, %c1_ui1 : !firrtl.uint, !firrtl.uint<1>
    %w1 = firrtl.wire  : !firrtl.uint<0>
    // CHECK: %0 = firrtl.tail %w, 1 : (!firrtl.uint<1>) -> !firrtl.uint<0>
    // CHECK: firrtl.connect %w1, %0 : !firrtl.uint<0>
    firrtl.connect %w1, %w : !firrtl.uint<0>, !firrtl.uint
  }
}