firrtl.circuit "AndOrXorReductionOp" {
  firrtl.module @AndOrXorReductionOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %3 = firrtl.andr {{.*}} -> !firrtl.uint<1>
    // CHECK: %4 = firrtl.orr {{.*}} -> !firrtl.uint<1>
    // CHECK: %5 = firrtl.xorr {{.*}} -> !firrtl.uint<1>
    %c0_ui16 = firrtl.constant 0 : !firrtl.uint<16>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.andr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    %4 = firrtl.orr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    %5 = firrtl.xorr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    firrtl.connect %0, %3 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %1, %4 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %2, %5 : !firrtl.uint, !firrtl.uint<1>
  }
}