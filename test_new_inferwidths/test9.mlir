firrtl.circuit "ConstCastOp" {
  firrtl.module @ConstCastOp() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.const.uint<1>
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %c1 = firrtl.constant 1 : !firrtl.const.uint<2>
    %c2 = firrtl.constant 2 : !firrtl.const.sint<3>
    %3 = firrtl.constCast %c1 : (!firrtl.const.uint<2>) -> !firrtl.uint<2>
    %4 = firrtl.constCast %c2 : (!firrtl.const.sint<3>) -> !firrtl.sint<3>
    firrtl.connect %0, %3 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %4 : !firrtl.sint, !firrtl.sint<3>
  }
}