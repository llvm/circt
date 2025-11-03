firrtl.circuit "CastOp" {
  firrtl.module @CastOp() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.asSInt {{.*}} -> !firrtl.sint<2>
    // CHECK: %5 = firrtl.asUInt {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.wire : !firrtl.clock
    %3 = firrtl.wire : !firrtl.asyncreset
    %4 = firrtl.asSInt %0 : (!firrtl.uint) -> !firrtl.sint
    %5 = firrtl.asUInt %1 : (!firrtl.sint) -> !firrtl.uint
    %6 = firrtl.asUInt %2 : (!firrtl.clock) -> !firrtl.uint<1>
    %7 = firrtl.asUInt %3 : (!firrtl.asyncreset) -> !firrtl.uint<1>
    %8 = firrtl.asClock %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.clock
    %9 = firrtl.asAsyncReset %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }
}