firrtl.circuit "RegSimple" {
  firrtl.module @RegSimple(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<6>
    // CHECK: %1 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<6>
    %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %1 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.xor %1, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %3 : !firrtl.uint, !firrtl.uint
    firrtl.connect %2, %x : !firrtl.uint, !firrtl.uint<6>
  }
}