firrtl.circuit "TransparentOps" {
  firrtl.module @TransparentOps(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>) {
    %false = firrtl.constant 0 : !firrtl.uint<1>
    %true = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>

    // CHECK: %ui = firrtl.wire : !firrtl.uint<5>
    %ui = firrtl.wire : !firrtl.uint

    firrtl.printf %clk, %false, "foo" : !firrtl.clock, !firrtl.uint<1>
    firrtl.skip
    firrtl.stop %clk, %false, 0 : !firrtl.clock, !firrtl.uint<1>
    firrtl.when %a : !firrtl.uint<1> {
      firrtl.connect %ui, %c0_ui4 : !firrtl.uint, !firrtl.uint<4>
    } else  {
      firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    }
    firrtl.assert %clk, %true, %true, "foo" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.assume %clk, %true, %true, "foo" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.int.unclocked_assume %true, %true, "foo" : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.cover %clk, %true, %true, "foo" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  }
}