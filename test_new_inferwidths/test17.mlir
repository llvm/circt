firrtl.circuit "ShlShrOp" {
  firrtl.module @ShlShrOp() {
    // CHECK: %0 = firrtl.shl {{.*}} -> !firrtl.uint<8>
    // CHECK: %1 = firrtl.shl {{.*}} -> !firrtl.sint<8>
    // CHECK: %2 = firrtl.shr {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = firrtl.shr {{.*}} -> !firrtl.sint<2>
    // CHECK: %4 = firrtl.shr {{.*}} -> !firrtl.uint<0>
    // CHECK: %5 = firrtl.shr {{.*}} -> !firrtl.sint<1>
    %ui = firrtl.wire : !firrtl.uint
    %si = firrtl.wire : !firrtl.sint

    %0 = firrtl.shl %ui, 3 : (!firrtl.uint) -> !firrtl.uint
    %1 = firrtl.shl %si, 3 : (!firrtl.sint) -> !firrtl.sint
    %2 = firrtl.shr %ui, 3 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.shr %si, 3 : (!firrtl.sint) -> !firrtl.sint
    %4 = firrtl.shr %ui, 9 : (!firrtl.uint) -> !firrtl.uint
    %5 = firrtl.shr %si, 9 : (!firrtl.sint) -> !firrtl.sint

    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    %c0_si5 = firrtl.constant 0 : !firrtl.sint<5>
    firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %si, %c0_si5 : !firrtl.sint, !firrtl.sint<5>

    // CHECK: firrtl.connect %u0, %0 : !firrtl.uint<8>
    %u0 = firrtl.wire : !firrtl.uint
    firrtl.connect %u0, %0 : !firrtl.uint, !firrtl.uint
    // CHECK: firrtl.connect %s1, %1 : !firrtl.sint<8>
    %s1 = firrtl.wire : !firrtl.sint
    firrtl.connect %s1, %1 : !firrtl.sint, !firrtl.sint
    // CHECK: firrtl.connect %u2, %2 : !firrtl.uint<2>
    %u2 = firrtl.wire : !firrtl.uint
    firrtl.connect %u2, %2 : !firrtl.uint, !firrtl.uint
    // CHECK: firrtl.connect %s3, %3 : !firrtl.sint<2>
    %s3 = firrtl.wire : !firrtl.sint
    firrtl.connect %s3, %3 : !firrtl.sint, !firrtl.sint
    // CHECK: firrtl.connect %u4, %4 : !firrtl.uint<0>
    %u4 = firrtl.wire : !firrtl.uint
    firrtl.connect %u4, %4 : !firrtl.uint, !firrtl.uint
    // CHECK: firrtl.connect %s5, %5 : !firrtl.sint<1>
    %s5 = firrtl.wire : !firrtl.sint
    firrtl.connect %s5, %5 : !firrtl.sint, !firrtl.sint
  }
}