firrtl.circuit "BitsHeadTailPadOp" {
  firrtl.module @BitsHeadTailPadOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %3 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %8 = firrtl.tail {{.*}} -> !firrtl.uint<12>
    // CHECK: %9 = firrtl.tail {{.*}} -> !firrtl.uint<12>
    // CHECK: %10 = firrtl.pad {{.*}} -> !firrtl.uint<42>
    // CHECK: %11 = firrtl.pad {{.*}} -> !firrtl.sint<42>
    // CHECK: %12 = firrtl.pad {{.*}} -> !firrtl.uint<99>
    // CHECK: %13 = firrtl.pad {{.*}} -> !firrtl.sint<99>
    %ui = firrtl.wire : !firrtl.uint
    %si = firrtl.wire : !firrtl.sint
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.wire : !firrtl.uint

    %4 = firrtl.bits %ui 3 to 1 : (!firrtl.uint) -> !firrtl.uint<3>
    %5 = firrtl.bits %si 3 to 1 : (!firrtl.sint) -> !firrtl.uint<3>
    %6 = firrtl.head %ui, 5 : (!firrtl.uint) -> !firrtl.uint<5>
    %7 = firrtl.head %si, 5 : (!firrtl.sint) -> !firrtl.uint<5>
    %8 = firrtl.tail %ui, 30 : (!firrtl.uint) -> !firrtl.uint
    %9 = firrtl.tail %si, 30 : (!firrtl.sint) -> !firrtl.uint
    %10 = firrtl.pad %ui, 13 : (!firrtl.uint) -> !firrtl.uint
    %11 = firrtl.pad %si, 13 : (!firrtl.sint) -> !firrtl.sint
    %12 = firrtl.pad %ui, 99 : (!firrtl.uint) -> !firrtl.uint
    %13 = firrtl.pad %si, 99 : (!firrtl.sint) -> !firrtl.sint

    firrtl.connect %0, %4 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %1, %5 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %6 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %3, %7 : !firrtl.uint, !firrtl.uint<5>

    %c0_ui42 = firrtl.constant 0 : !firrtl.uint<42>
    %c0_si42 = firrtl.constant 0 : !firrtl.sint<42>
    firrtl.connect %ui, %c0_ui42 : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %si, %c0_si42 : !firrtl.sint, !firrtl.sint<42>
  }
}