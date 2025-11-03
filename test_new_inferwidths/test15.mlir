firrtl.circuit "MuxOp" {
  firrtl.module @MuxOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<0>
    // CHECK: %3 = firrtl.mux{{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.mux(%2, %0, %1) : (!firrtl.uint, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %c0_ui0 : !firrtl.uint, !firrtl.uint<0>
  }
}