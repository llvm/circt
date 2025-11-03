firrtl.circuit "PassiveCastOp" {
  firrtl.module @PassiveCastOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %1 = builtin.unrealized_conversion_cast %ui : !firrtl.uint<5> to !firrtl.uint<5>
    %ui = firrtl.wire : !firrtl.uint
    %0 = firrtl.wire : !firrtl.uint
    %1 = builtin.unrealized_conversion_cast %ui : !firrtl.uint to !firrtl.uint
    firrtl.connect %0, %1 : !firrtl.uint, !firrtl.uint
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
  }
}