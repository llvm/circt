// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-imconstprop)' --split-input-file  %s | FileCheck %s

firrtl.circuit "Test" {

  // CHECK-LABEL: @PassThrough
  // CHECK: (in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>)
  firrtl.module @PassThrough(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>) {
    // CHECK-NEXT: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    %dontTouchWire = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK-NEXT: %dontTouchWire = firrtl.wire
    firrtl.connect %dontTouchWire, %source : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: firrtl.connect %dontTouchWire, %c0_ui1

    // CHECK-NEXT: firrtl.connect %dest, %dontTouchWire
    firrtl.connect %dest, %dontTouchWire : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: @Test
  firrtl.module @Test(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                      out %result1: !firrtl.uint<1>,
                      out %result2: !firrtl.clock,
                      out %result3: !firrtl.uint<1>,
                      out %result4: !firrtl.uint<1>,
                      out %result5: !firrtl.uint<2>,
                      out %result6: !firrtl.uint<2>,
                      out %result7: !firrtl.uint<4>,
                      out %result8: !firrtl.uint<4>,
                      out %result9: !firrtl.uint<4>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // Trivial wire constant propagation.
    %someWire = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %someWire, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK-NOT: firrtl.wire
    // CHECK: firrtl.connect %result1, %c0_ui1_0
    firrtl.connect %result1, %someWire : !firrtl.uint<1>, !firrtl.uint<1>

    // Trivial wire special constant propagation.
    %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
    %clockWire = firrtl.wire : !firrtl.clock
    firrtl.connect %clockWire, %c0_clock : !firrtl.clock, !firrtl.clock

    // CHECK-NOT: firrtl.wire
    // CHECK: firrtl.connect %result2, %c0_clock
    firrtl.connect %result2, %clockWire : !firrtl.clock, !firrtl.clock


    // Not a constant.
    %nonconstWire = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %nonconstWire, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %nonconstWire, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: firrtl.connect %result3, %nonconstWire
    firrtl.connect %result3, %nonconstWire : !firrtl.uint<1>, !firrtl.uint<1>


    // Constant propagation through instance.
    %source, %dest = firrtl.instance @PassThrough {name = "", portNames = ["source", "dest"]} : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: firrtl.connect %inst_source, %c0_ui1
    firrtl.connect %source, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
<<<<<<< HEAD
    // CHECK: firrtl.connect %result3, %inst_dest
    firrtl.connect %result3, %dest : !firrtl.uint<1>, !firrtl.uint<1>
||||||| parent of f3ea2e7d ([FIRRTL] Allow firrtl.constant of type Clock, Reset, AsyncReset)
    // CHECK: firrtl.connect %result3, %c0_ui1_1
    firrtl.connect %result3, %dest : !firrtl.uint<1>, !firrtl.uint<1>
=======
    // CHECK: firrtl.connect %result4, %c0_ui1_1
    firrtl.connect %result4, %dest : !firrtl.uint<1>, !firrtl.uint<1>
>>>>>>> f3ea2e7d ([FIRRTL] Allow firrtl.constant of type Clock, Reset, AsyncReset)

    // Check connect extensions.
    %extWire = firrtl.wire : !firrtl.uint<2>
    firrtl.connect %extWire, %c0_ui1 : !firrtl.uint<2>, !firrtl.uint<1>

    // Connects of invalid values shouldn't hurt.
    %invalid = firrtl.invalidvalue : !firrtl.uint<2>
    firrtl.connect %extWire, %invalid : !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK: firrtl.connect %result5, %c0_ui2
    firrtl.connect %result5, %extWire: !firrtl.uint<2>, !firrtl.uint<2>

    // regreset
    %c0_ui20 = firrtl.constant 0 : !firrtl.uint<20>
    %regreset = firrtl.regreset %clock, %reset, %c0_ui20  : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<20>) -> !firrtl.uint<2>

    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    firrtl.connect %regreset, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK: firrtl.connect %result6, %c0_ui2
    firrtl.connect %result6, %regreset: !firrtl.uint<2>, !firrtl.uint<2>

    // reg
    %reg = firrtl.reg %clock  : (!firrtl.clock) -> !firrtl.uint<4>
    firrtl.connect %reg, %c0_ui2 : !firrtl.uint<4>, !firrtl.uint<2>
    // CHECK: firrtl.connect %result7, %c0_ui4
    firrtl.connect %result7, %reg: !firrtl.uint<4>, !firrtl.uint<4>

    // Wire without connects to it should turn into 'invalid'.
    %unconnectedWire = firrtl.wire : !firrtl.uint<2>
    // CHECK: firrtl.connect %result8, %invalid_ui2
    firrtl.connect %result8, %unconnectedWire: !firrtl.uint<4>, !firrtl.uint<2>

    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>

    // Multiple operations that fold to constants shouldn't leave dead constants
    // around.
    %a = firrtl.and %extWire, %c2_ui2 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    %b = firrtl.or %a, %c1_ui2 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    // CHECK-NEXT: firrtl.constant 3
    %c = firrtl.xor %b, %c2_ui2 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>

    // CHECK-NEXT: firrtl.connect %result9, %c3_ui2
    firrtl.connect %result9, %c: !firrtl.uint<4>, !firrtl.uint<2>


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

    %1 = firrtl.subfield %0(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.sint<8>
    %2 = firrtl.subfield %0(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.uint<4>
    firrtl.connect %2, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
    %3 = firrtl.subfield %0(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.uint<1>
    firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = firrtl.subfield %0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.clock
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
<<<<<<< HEAD
}

// -----
// DontTouch annotation should block constant propagation.
firrtl.circuit "testDontTouch"  {
  // CHECK-LABEL: firrtl.module @blockProp 
  firrtl.module @blockProp1(in %clock: !firrtl.clock, in %a: !firrtl.uint<1> {firrtl.annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}, out %b: !firrtl.uint<1>) {
    //CHECK: %c = firrtl.reg 
    %c = firrtl.reg %clock  : (!firrtl.clock) -> !firrtl.uint<1>
    firrtl.connect %c, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %c : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @allowProp 
  firrtl.module @allowProp(in %clock: !firrtl.clock, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // CHECK: [[CONST:%.+]] = firrtl.constant 1 : !firrtl.uint<1>
    %c = firrtl.reg %clock  : (!firrtl.clock) -> !firrtl.uint<1>
    firrtl.connect %c, %a : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %b, [[CONST]] 
    firrtl.connect %b, %c : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @blockProp3
  firrtl.module @blockProp3(in %clock: !firrtl.clock, in %a: !firrtl.uint<1> , out %b: !firrtl.uint<1>) {
    //CHECK: %c = firrtl.reg
    %c = firrtl.reg %clock {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : (!firrtl.clock) -> !firrtl.uint<1>
    firrtl.connect %c, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %c : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @testDontTouch
  firrtl.module @testDontTouch(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>, out %a1: !firrtl.uint<1>, out %a2: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %blockProp1_clock, %blockProp1_a, %blockProp1_b = firrtl.instance @blockProp1  {name = "blockProp1"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    %allowProp_clock, %allowProp_a, %allowProp_b = firrtl.instance @allowProp  {name = "allowProp"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    %blockProp3_clock, %blockProp3_a, %blockProp3_b = firrtl.instance @blockProp3  {name = "blockProp3"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %blockProp1_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %allowProp_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %blockProp3_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %blockProp1_a, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %allowProp_a, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %blockProp3_a, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %a, %blockProp1_b
    firrtl.connect %a, %blockProp1_b : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %a1, %c
    firrtl.connect %a1, %allowProp_b : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %a2, %blockProp3_b
    firrtl.connect %a2, %blockProp3_b : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @CheckNode
  firrtl.module @CheckNode(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    %z = firrtl.node %x  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK: firrtl.connect %y, %z
    firrtl.connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }

}

firrtl.circuit "OutPortTop" {
    firrtl.module @OutPortChild1(out %out: !firrtl.uint<1> {firrtl.annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}) {
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      firrtl.connect %out, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    }
    firrtl.module @OutPortChild2(out %out: !firrtl.uint<1>) {
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      firrtl.connect %out, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    }
  // CHECK-LABEL: firrtl.module @OutPortTop
    firrtl.module @OutPortTop(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
      %c_out = firrtl.instance @OutPortChild1  {name = "c"} : !firrtl.uint<1>
      %c_out_0 = firrtl.instance @OutPortChild2  {name = "c"} : !firrtl.uint<1>
      // CHECK: %0 = firrtl.and %x, %c_out
      %0 = firrtl.and %x, %c_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK: %1 = firrtl.and %0, %c0_ui1
      %1 = firrtl.and %0, %c_out_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %z, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    }
}

firrtl.circuit "InputPortTop"   {
  // CHECK-LABEL: firrtl.module @InputPortChild2
  firrtl.module @InputPortChild2(in %in0: !firrtl.uint<1>, in %in1: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    // CHECK: = firrtl.constant 1
    %0 = firrtl.and %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @InputPortChild
  firrtl.module @InputPortChild(in %in0: !firrtl.uint<1>, in %in1: !firrtl.uint<1> {firrtl.annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}, out %out: !firrtl.uint<1>) {
    // CHECK: %0 = firrtl.and %in0, %in1
    %0 = firrtl.and %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @InputPortTop(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>, out %z2: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c_in0, %c_in1, %c_out = firrtl.instance @InputPortChild  {name = "c"} : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    %c2_in0, %c2_in1, %c2_out = firrtl.instance @InputPortChild2  {name = "c2"} : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z, %c_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c_in0, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c_in1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %z2, %c2_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c2_in0, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c2_in1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
||||||| parent of f3ea2e7d ([FIRRTL] Allow firrtl.constant of type Clock, Reset, AsyncReset)
}
=======
}
>>>>>>> f3ea2e7d ([FIRRTL] Allow firrtl.constant of type Clock, Reset, AsyncReset)
