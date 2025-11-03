firrtl.circuit "Issue1088" {
  firrtl.module @Issue1088(out %y: !firrtl.sint<4>) {
    // CHECK: %x = firrtl.wire : !firrtl.sint<9>
    // CHECK: %c200_si9 = firrtl.constant 200 : !firrtl.sint<9>
    // CHECK: %0 = firrtl.tail %x, 5 : (!firrtl.sint<9>) -> !firrtl.uint<4>
    // CHECK: %1 = firrtl.asSInt %0 : (!firrtl.uint<4>) -> !firrtl.sint<4>
    // CHECK: firrtl.connect %y, %1 : !firrtl.sint<4>
    // CHECK: firrtl.connect %x, %c200_si9 : !firrtl.sint<9>
    %x = firrtl.wire : !firrtl.sint
    %c200_si = firrtl.constant 200 : !firrtl.sint
    firrtl.connect %y, %x : !firrtl.sint<4>, !firrtl.sint
    firrtl.connect %x, %c200_si : !firrtl.sint, !firrtl.sint
  }
}