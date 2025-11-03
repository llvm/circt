firrtl.circuit "Issue1271" {
  firrtl.module @Issue1271(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>) {
    // CHECK: %a = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<2>
    // CHECK: %b = firrtl.node %0  : !firrtl.uint<3>
    // CHECK: %c = firrtl.node %1  : !firrtl.uint<2>
    %a = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.add %a, %c0_ui1 : (!firrtl.uint, !firrtl.uint<1>) -> !firrtl.uint
    %b = firrtl.node %0  : !firrtl.uint
    %1 = firrtl.tail %b, 1 : (!firrtl.uint) -> !firrtl.uint
    %c = firrtl.node %1  : !firrtl.uint
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %2 = firrtl.mux(%cond, %c0_ui2, %c) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %a, %2 : !firrtl.uint, !firrtl.uint
  }
}