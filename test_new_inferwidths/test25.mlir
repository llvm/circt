firrtl.circuit "RegResetSimple" {
  firrtl.module @RegResetSimple(
    in %clk: !firrtl.clock,
    in %rst: !firrtl.asyncreset,
    in %x: !firrtl.uint<6>
  ) {
    // CHECK: %0 = firrtl.regreset %clk, %rst, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<6>
    // CHECK: %1 = firrtl.regreset %clk, %rst, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<6>
    // CHECK: %2:2 = firrtl.regreset %clk, %rst, %c0_ui17 forceable : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint<17>, !firrtl.rwprobe<uint<17>>
    // CHECK: %3 = firrtl.regreset %clk, %rst, %c0_ui17 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint<17>
    %c0_ui = firrtl.constant 0 : !firrtl.uint
    %c0_ui17 = firrtl.constant 0 : !firrtl.uint<17>
    %0 = firrtl.regreset %clk, %rst, %c0_ui : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint, !firrtl.uint
    %1 = firrtl.regreset %clk, %rst, %c0_ui : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint, !firrtl.uint
    %2:2 = firrtl.regreset %clk, %rst, %c0_ui17 forceable : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint, !firrtl.rwprobe<uint>
    %3 = firrtl.regreset %clk, %rst, %c0_ui17 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint
    %4 = firrtl.wire : !firrtl.uint
    %5 = firrtl.xor %1, %4 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %5 : !firrtl.uint, !firrtl.uint
    firrtl.connect %2, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %3, %5 : !firrtl.uint, !firrtl.uint
    firrtl.connect %4, %x : !firrtl.uint, !firrtl.uint<6>
  }
}