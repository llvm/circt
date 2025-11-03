firrtl.circuit "NotOp" {
  firrtl.module @NotOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %2 = firrtl.not {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = firrtl.not {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.not %0 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.not %1 : (!firrtl.sint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }
}