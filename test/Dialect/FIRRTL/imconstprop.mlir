// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-imconstprop)' --split-input-file  %s | FileCheck %s

firrtl.circuit "Test" {

  // CHECK-LABEL: @PassThrough
  // CHECK: (in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>)
  firrtl.module @PassThrough(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>) {
    // CHECK-NEXT: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-NEXT: %c0_ui1_0 = firrtl.constant 0 : !firrtl.uint<1>

    %dontTouchWire = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK-NEXT: %dontTouchWire = firrtl.wire
    firrtl.connect %dontTouchWire, %source : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: firrtl.connect %dontTouchWire, %c0_ui1

    // CHECK-NEXT: firrtl.connect %dest, %c0_ui1_0
    firrtl.connect %dest, %dontTouchWire : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: @Test
  firrtl.module @Test(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                      out %result1: !firrtl.uint<1>,
                      out %result2: !firrtl.uint<1>,
                      out %result3: !firrtl.uint<1>,
                      out %result4: !firrtl.uint<2>,
                      out %result5: !firrtl.uint<2>,
                      out %result6: !firrtl.uint<4>,
                      out %result7: !firrtl.uint<4>,
                      out %result8: !firrtl.uint<4>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

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
    %source, %dest = firrtl.instance @PassThrough {name = "", portNames = ["source", "dest"]} : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: firrtl.connect %inst_source, %c0_ui1
    firrtl.connect %source, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
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
    %c0_ui20 = firrtl.constant 0 : !firrtl.uint<20>
    %regreset = firrtl.regreset %clock, %reset, %c0_ui20  : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<20>) -> !firrtl.uint<2>

    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    firrtl.connect %regreset, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK: firrtl.connect %result5, %c0_ui2
    firrtl.connect %result5, %regreset: !firrtl.uint<2>, !firrtl.uint<2>

    // reg
    %reg = firrtl.reg %clock  : (!firrtl.clock) -> !firrtl.uint<4>
    firrtl.connect %reg, %c0_ui2 : !firrtl.uint<4>, !firrtl.uint<2>
    // CHECK: firrtl.connect %result6, %c0_ui4
    firrtl.connect %result6, %reg: !firrtl.uint<4>, !firrtl.uint<4>

    // Wire without connects to it should turn into 'invalid'.
    %unconnectedWire = firrtl.wire : !firrtl.uint<2>
    // CHECK: firrtl.connect %result7, %invalid_ui2
    firrtl.connect %result7, %unconnectedWire: !firrtl.uint<4>, !firrtl.uint<2>

    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>

    // Multiple operations that fold to constants shouldn't leave dead constants
    // around.
    %a = firrtl.and %extWire, %c2_ui2 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    %b = firrtl.or %a, %c1_ui2 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    // CHECK-NEXT: firrtl.constant 3
    %c = firrtl.xor %b, %c2_ui2 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>

    // CHECK-NEXT: firrtl.connect %result8, %c3_ui2
    firrtl.connect %result8, %c: !firrtl.uint<4>, !firrtl.uint<2>


    // Constant propagation through instance.
    firrtl.instance @ReadMem {name = "ReadMem"}
  }

  // Unused modules should NOT be completely dropped.
  // https://github.com/llvm/circt/issues/1236

  // CHECK-LABEL: @UnusedModule(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>)
  firrtl.module @UnusedModule(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>) {
    // CHECK-NEXT: firrtl.connect %dest, %source
    firrtl.connect %dest, %source : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: }
  }


  // CHECK-LABEL: ReadMem
  firrtl.module @ReadMem() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    %0 = firrtl.mem Undefined {depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>

    %1 = firrtl.subfield %0("data") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.sint<8>
    %2 = firrtl.subfield %0("addr") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.uint<4>
    firrtl.connect %2, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
    %3 = firrtl.subfield %0("en") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.uint<1>
    firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = firrtl.subfield %0("clk") : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.clock
  }
}

// -----
firrtl.circuit "RegParent" {
  /// Properly handle reset types as constants.
  firrtl.module @RegParent(in %clock: !firrtl.clock, out %out: !firrtl.uint<1>) {
    %x_in, %x_clock, %x_out = firrtl.instance @RegChild  {name = "x"} : !firrtl.reset, !firrtl.clock, !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.connect %x_in, %c0_ui1 : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %x_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %out, %x_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: @RegChild
  firrtl.module @RegChild(in %in: !firrtl.reset, in %clock: !firrtl.clock, out %out: !firrtl.uint<1>) {
    // CHECK-NOT: %arst = firrtl.wire
    %arst = firrtl.wire  : !firrtl.reset
    firrtl.connect %arst, %in : !firrtl.reset, !firrtl.reset
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // Register should have a constant reset value.
    // CHECK: %c0_reset = firrtl.constant 0 : !firrtl.reset
    // CHECK: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %myreg = firrtl.regreset %clock, %c0_reset, %c0_ui1  : (!firrtl.clock, !firrtl.reset, !firrtl.uint<1>) -> !firrtl.uint<1>
    %myreg = firrtl.regreset %clock, %arst, %c0_ui1  : (!firrtl.clock, !firrtl.reset, !firrtl.uint<1>) -> !firrtl.uint<1>
    // Don't optimize away the register..
    %0 = firrtl.asUInt %clock : (!firrtl.clock) -> !firrtl.uint<1>
    %1 = firrtl.not %myreg : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %2 = firrtl.mux(%0, %1, %myreg) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %myreg, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %myreg : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: firrtl.module @Issue1188
// https://github.com/llvm/circt/issues/1188
// Make sure that we handle recursion through muxes correctly.
firrtl.circuit "Issue1188"  {
  firrtl.module @Issue1188(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %io_out: !firrtl.uint<6>, out %io_out3: !firrtl.uint<3>) {
    %c1_ui6 = firrtl.constant 1 : !firrtl.uint<6>
    %D0123456 = firrtl.reg %clock  : (!firrtl.clock) -> !firrtl.uint<6>
    %0 = firrtl.bits %D0123456 4 to 0 : (!firrtl.uint<6>) -> !firrtl.uint<5>
    %1 = firrtl.bits %D0123456 5 to 5 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %2 = firrtl.cat %0, %1 : (!firrtl.uint<5>, !firrtl.uint<1>) -> !firrtl.uint<6>
    %3 = firrtl.bits %D0123456 4 to 4 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %4 = firrtl.xor %2, %3 : (!firrtl.uint<6>, !firrtl.uint<1>) -> !firrtl.uint<6>
    %5 = firrtl.bits %D0123456 1 to 1 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %6 = firrtl.bits %D0123456 3 to 3 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %7 = firrtl.cat %5, %6 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    %8 = firrtl.cat %7, %1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<3>
    firrtl.connect %io_out, %D0123456 : !firrtl.uint<6>, !firrtl.uint<6>
    firrtl.connect %io_out3, %8 : !firrtl.uint<3>, !firrtl.uint<3>
    // CHECK: firrtl.mux(%reset, %c1_ui6, %4)
    %9 = firrtl.mux(%reset, %c1_ui6, %4) : (!firrtl.uint<1>, !firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    firrtl.connect %D0123456, %9 : !firrtl.uint<6>, !firrtl.uint<6>
  }
}
