// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-imconstprop)' --split-input-file  %s | FileCheck %s

; XFAIL: *
//propagate module chains not connected to the top
firrtl.circuit "TopUnconnected"   {
  firrtl.module @TopUnconnected(in %dummy: !firrtl.uint<1>) {
  }
  // CHECK-LABEL: firrtl.module @unconnectedChild1
  firrtl.module @unconnectedChild1(out %out: !firrtl.uint<1>) {
    %one_test = firrtl.instance @baz1  {name = "one"} : !firrtl.uint<1>
    %zero_test = firrtl.instance @baz0  {name = "zero"} : !firrtl.uint<1>
    %0 = firrtl.or %one_test, %zero_test : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK:  firrtl.connect %out, %c1_ui1_0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @bar0(out %out: !firrtl.uint<1>) {
    %one_test = firrtl.instance @baz1  {name = "one"} : !firrtl.uint<1>
    %zero_test = firrtl.instance @baz0  {name = "zero"} : !firrtl.uint<1>
    %0 = firrtl.and %one_test, %zero_test : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @baz1(out %test: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %test, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @baz0(out %test: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.connect %test, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

//"ConstProp" should "NOT touch self-inits"
firrtl.circuit "selfInit"   {
  // CHECK-LABEL: firrtl.module @selfInit
  firrtl.module @selfInit(in %clk: !firrtl.clock, in %rst: !firrtl.uint<1>, out %z: !firrtl.uint<4>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %selfinit = firrtl.reg %clk  : (!firrtl.clock) -> !firrtl.uint<1>
    // CHECK: firrtl.connect %selfinit, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %selfinit, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %0 = firrtl.mux(%c1_ui, %c0_ui2, %c0_ui4) : (!firrtl.uint, !firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>
    firrtl.connect %z, %0 : !firrtl.uint<4>, !firrtl.uint<4>
  }
}

//"Registers with constant reset and connection to the same constant" should "be replaced with that constant"
firrtl.circuit "regConstReset"   {
  firrtl.module @regConstReset(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %r1 = firrtl.reg %clock  : (!firrtl.clock) -> !firrtl.uint<1>
    %r2 = firrtl.reg %clock  : (!firrtl.clock) -> !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.mux(%en, %c1_ui1, %r1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %r1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.mux(%en, %r2, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %r2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.xor %r1, %r2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: firrtl.connect %out, %c
    firrtl.connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

//"Const prop of registers" should "do limited speculative expansion of optimized muxes to absorb bigger cones"
firrtl.circuit "constPropRegMux"   {
  firrtl.module @constPropRegMux(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cmd: !firrtl.uint<3>, out %z: !firrtl.uint<8>) {
    %c7_ui4 = firrtl.constant 7 : !firrtl.uint<4>
    %r = firrtl.regreset %clock, %reset, %c7_ui4  : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>) -> !firrtl.uint<8>
    %c0_ui3 = firrtl.constant 0 : !firrtl.uint<3>
    %0 = firrtl.eq %cmd, %c0_ui3 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %c7_ui3 = firrtl.constant 7 : !firrtl.uint<3>
    %c1_ui3 = firrtl.constant 1 : !firrtl.uint<3>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %1 = firrtl.not %0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %2 = firrtl.eq %cmd, %c1_ui3 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %3 = firrtl.and %1, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %4 = firrtl.not %2 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %5 = firrtl.and %1, %4 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %6 = firrtl.eq %cmd, %c2_ui3 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %7 = firrtl.and %5, %6 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %8 = firrtl.not %6 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %9 = firrtl.and %5, %8 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = firrtl.mux(%7, %c7_ui4, %r) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<8>) -> !firrtl.uint<8>
    %11 = firrtl.mux(%3, %r, %10) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    %12 = firrtl.mux(%0, %c7_ui3, %11) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.connect %r, %12 : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
  }
}
