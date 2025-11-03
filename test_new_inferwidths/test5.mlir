firrtl.circuit "AndOrXorOp" {
  firrtl.module @AndOrXorOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.and {{.*}} -> !firrtl.uint<3>
    // CHECK: %3 = firrtl.or {{.*}} -> !firrtl.uint<3>
    // CHECK: %4 = firrtl.xor {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.and %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %3 = firrtl.or %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %4 = firrtl.xor %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }
}