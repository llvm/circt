firrtl.circuit "RegShl" {
  firrtl.module @RegShl(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<6>
    %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %1 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %2 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %3 = firrtl.shl %0, 0 : (!firrtl.uint) -> !firrtl.uint
    %4 = firrtl.shl %1, 3 : (!firrtl.uint) -> !firrtl.uint
    %5 = firrtl.shr %4, 3 : (!firrtl.uint) -> !firrtl.uint
    %6 = firrtl.shr %1, 3 : (!firrtl.uint) -> !firrtl.uint
    %7 = firrtl.shl %6, 3 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %2, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %0, %2 : !firrtl.uint, !firrtl.uint
    firrtl.connect %1, %5 : !firrtl.uint, !firrtl.uint
    firrtl.connect %2, %7 : !firrtl.uint, !firrtl.uint
  }
}