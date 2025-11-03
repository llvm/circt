firrtl.circuit "CatDynShiftOp" {
  firrtl.module @CatDynShiftOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.sint<2>
    // CHECK: %3 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.cat {{.*}} -> !firrtl.uint<8>
    // CHECK: %5 = firrtl.cat {{.*}} -> !firrtl.uint<8>
    // CHECK: %6 = firrtl.dshl {{.*}} -> !firrtl.uint<10>
    // CHECK: %7 = firrtl.dshl {{.*}} -> !firrtl.sint<10>
    // CHECK: %8 = firrtl.dshlw {{.*}} -> !firrtl.uint<3>
    // CHECK: %9 = firrtl.dshlw {{.*}} -> !firrtl.sint<3>
    // CHECK: %10 = firrtl.dshr {{.*}} -> !firrtl.uint<3>
    // CHECK: %11 = firrtl.dshr {{.*}} -> !firrtl.sint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.sint
    %3 = firrtl.wire : !firrtl.sint
    %4 = firrtl.cat %0, %1, %1 : (!firrtl.uint, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %5 = firrtl.cat %2, %3, %3 : (!firrtl.sint, !firrtl.sint, !firrtl.sint) -> !firrtl.uint
    %6 = firrtl.dshl %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %7 = firrtl.dshl %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %8 = firrtl.dshlw %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %9 = firrtl.dshlw %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %10 = firrtl.dshr %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %11 = firrtl.dshr %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %c1_si2 = firrtl.constant 1 : !firrtl.sint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %c1_si2 : !firrtl.sint, !firrtl.sint<2>
    firrtl.connect %3, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }
}