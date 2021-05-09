// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-imconstprop)' %s | FileCheck %s

firrtl.circuit "Test" {

  // CHECK-LABEL: @PassThrough
  // CHECK: (in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>)
  firrtl.module @PassThrough(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>) {
    // CHECK-NEXT: %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.connect %dest, %c0_ui1
    firrtl.connect %dest, %source : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: @Test
  firrtl.module @Test(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                      out %result1: !firrtl.uint<1>,
                      out %result2: !firrtl.uint<1>,
                      out %result3: !firrtl.uint<1>,
                      out %result4: !firrtl.uint<2>,
                      out %result5: !firrtl.uint<2>,
                      out %result6: !firrtl.uint<4>) {
    %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>

    // Trivial wire constant propagation.
    %someWire = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %someWire, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK-NOT: firrtl.wire
    // CHECK: firrtl.connect %result1, %c0_ui1_0
    firrtl.connect %result1, %someWire : !firrtl.uint<1>, !firrtl.uint<1>

    // Not a constant.
    %nonconstWire = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %nonconstWire, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %nonconstWire, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: firrtl.connect %result2, %nonconstWire
    firrtl.connect %result2, %nonconstWire : !firrtl.uint<1>, !firrtl.uint<1>


    // Constant propagation through instance.
    %source, %dest = firrtl.instance @PassThrough {name = "", portNames = ["source", "dest"]} : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    // CHECK: firrtl.connect %inst_source, %c0_ui1
    firrtl.connect %source, %c0_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    // CHECK: firrtl.connect %result3, %c0_ui1_1
    firrtl.connect %result3, %dest : !firrtl.uint<1>, !firrtl.uint<1>

    // Check connect extensions.
    %extWire = firrtl.wire : !firrtl.uint<2>
    firrtl.connect %extWire, %c0_ui1 : !firrtl.uint<2>, !firrtl.uint<1>

    // Connects of invalid values shouldn't hurt.
    %invalid = firrtl.invalidvalue : !firrtl.uint<2>
    firrtl.connect %extWire, %invalid : !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK: firrtl.connect %result4, %c0_ui2
    firrtl.connect %result4, %extWire: !firrtl.uint<2>, !firrtl.uint<2>

    // regreset
    %c0_ui20 = firrtl.constant(0 : ui20) : !firrtl.uint<20>
    %regreset = firrtl.regreset %clock, %reset, %c0_ui20  : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<20>) -> !firrtl.uint<2>

    %c0_ui2 = firrtl.constant(0 : ui2) : !firrtl.uint<2>
    firrtl.connect %regreset, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK: firrtl.connect %result5, %c0_ui2
    firrtl.connect %result5, %regreset: !firrtl.uint<2>, !firrtl.uint<2>

    // reg
    %reg = firrtl.reg %clock  : (!firrtl.clock) -> !firrtl.uint<4>
    firrtl.connect %reg, %c0_ui2 : !firrtl.uint<4>, !firrtl.uint<2>

    // CHECK: firrtl.connect %result6, %c0_ui4
    firrtl.connect %result6, %reg: !firrtl.uint<4>, !firrtl.uint<4>

  }

  // Unused modules should be completely dropped.

  // CHECK-LABEL: @UnusedModule(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>)
  firrtl.module @UnusedModule(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>) {
    firrtl.connect %dest, %source : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: }
  }
}
