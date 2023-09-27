// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop))' --split-input-file  %s | FileCheck %s

firrtl.circuit "Test" {

  // CHECK-LABEL: @PassThrough
  // CHECK: (in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>)
  firrtl.module private @PassThrough(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>) {
    // CHECK-NEXT: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    %dontTouchWire = firrtl.wire sym @a1 : !firrtl.uint<1>
    // CHECK-NEXT: %dontTouchWire = firrtl.wire
    firrtl.strictconnect %dontTouchWire, %source : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %dontTouchWire, %c0_ui1

    // CHECK-NEXT: firrtl.strictconnect %dest, %dontTouchWire
    firrtl.strictconnect %dest, %dontTouchWire : !firrtl.uint<1>
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
                      out %result9: !firrtl.uint<2>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // Trivial wire constant propagation.
    %someWire = firrtl.wire interesting_name : !firrtl.uint<1>
    firrtl.strictconnect %someWire, %c0_ui1 : !firrtl.uint<1>

    // CHECK: %someWire = firrtl.wire
    // CHECK: firrtl.strictconnect %someWire, %c0_ui1
    // CHECK: firrtl.strictconnect %result1, %c0_ui1
    firrtl.strictconnect %result1, %someWire : !firrtl.uint<1>

    // Trivial wire special constant propagation.
    %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
    %clockWire = firrtl.wire interesting_name : !firrtl.clock
    firrtl.strictconnect %clockWire, %c0_clock : !firrtl.clock

    // CHECK: %clockWire = firrtl.wire
    // CHECK: firrtl.strictconnect %clockWire, %c0_clock
    // CHECK: firrtl.strictconnect %result2, %c0_clock
    firrtl.strictconnect %result2, %clockWire : !firrtl.clock

    // Not a constant.
    %nonconstWire = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %nonconstWire, %c0_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %nonconstWire, %c1_ui1 : !firrtl.uint<1>

    // CHECK: firrtl.strictconnect %result3, %nonconstWire
    firrtl.strictconnect %result3, %nonconstWire : !firrtl.uint<1>

    // Constant propagation through instance.
    %source, %dest = firrtl.instance "" sym @dm21 @PassThrough(in source: !firrtl.uint<1>, out dest: !firrtl.uint<1>)

    // CHECK: firrtl.strictconnect %inst_source, %c0_ui1
    firrtl.strictconnect %source, %c0_ui1 : !firrtl.uint<1>

    // CHECK: firrtl.strictconnect %result4, %inst_dest
    firrtl.strictconnect %result4, %dest : !firrtl.uint<1>

    // Check connect extensions.
    %extWire = firrtl.wire : !firrtl.uint<2>
    firrtl.strictconnect %extWire, %c0_ui2 : !firrtl.uint<2>

    // Connects of invalid values should hurt.
    %invalid = firrtl.invalidvalue : !firrtl.uint<2>
    firrtl.strictconnect %extWire, %invalid : !firrtl.uint<2>

    // CHECK-NOT: firrtl.strictconnect %result5, %c0_ui2
    firrtl.strictconnect %result5, %extWire: !firrtl.uint<2>

    // Constant propagation through instance.
    firrtl.instance ReadMem @ReadMem()
  }

  // Unused modules should NOT be completely dropped.
  // https://github.com/llvm/circt/issues/1236

  // CHECK-LABEL: @UnusedModule(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>)
  firrtl.module private @UnusedModule(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>) {
    // CHECK-NEXT: firrtl.strictconnect %dest, %source
    firrtl.strictconnect %dest, %source : !firrtl.uint<1>
    // CHECK-NEXT: }
  }


  // CHECK-LABEL: ReadMem
  firrtl.module private @ReadMem() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<4>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    %0 = firrtl.mem Undefined {depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>

    %1 = firrtl.subfield %0[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    %2 = firrtl.subfield %0[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    firrtl.strictconnect %2, %c0_ui1 : !firrtl.uint<4>
    %3 = firrtl.subfield %0[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    firrtl.strictconnect %3, %c1_ui1 : !firrtl.uint<1>
    %4 = firrtl.subfield %0[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
  }
}

// -----

// CHECK-LABEL: firrtl.module @Issue1188
// https://github.com/llvm/circt/issues/1188
// Make sure that we handle recursion through muxes correctly.
firrtl.circuit "Issue1188"  {
  firrtl.module @Issue1188(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %io_out: !firrtl.uint<6>, out %io_out3: !firrtl.uint<3>) {
    %c1_ui6 = firrtl.constant 1 : !firrtl.uint<6>
    %D0123456 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<6>
    %0 = firrtl.bits %D0123456 4 to 0 : (!firrtl.uint<6>) -> !firrtl.uint<5>
    %1 = firrtl.bits %D0123456 5 to 5 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %2 = firrtl.cat %0, %1 : (!firrtl.uint<5>, !firrtl.uint<1>) -> !firrtl.uint<6>
    %3 = firrtl.bits %D0123456 4 to 4 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %4 = firrtl.xor %2, %3 : (!firrtl.uint<6>, !firrtl.uint<1>) -> !firrtl.uint<6>
    %5 = firrtl.bits %D0123456 1 to 1 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %6 = firrtl.bits %D0123456 3 to 3 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %7 = firrtl.cat %5, %6 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    %8 = firrtl.cat %7, %1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<3>
    firrtl.strictconnect %io_out, %D0123456 : !firrtl.uint<6>
    firrtl.strictconnect %io_out3, %8 : !firrtl.uint<3>
    // CHECK: firrtl.mux(%reset, %c1_ui6, %4)
    %9 = firrtl.mux(%reset, %c1_ui6, %4) : (!firrtl.uint<1>, !firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    firrtl.strictconnect %D0123456, %9 : !firrtl.uint<6>
  }
}

// -----

// DontTouch annotation should block constant propagation.
firrtl.circuit "testDontTouch"  {
  // CHECK-LABEL: firrtl.module private @blockProp
  firrtl.module private @blockProp1(in %clock: !firrtl.clock,
    in %a: !firrtl.uint<1> sym @dntSym, out %b: !firrtl.uint<1>){
    //CHECK: %c = firrtl.reg
    %c = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    firrtl.strictconnect %c, %a : !firrtl.uint<1>
    firrtl.strictconnect %b, %c : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module private @allowProp
  firrtl.module private @allowProp(in %clock: !firrtl.clock, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // CHECK: [[CONST:%.+]] = firrtl.constant 1 : !firrtl.uint<1>
    %c = firrtl.wire  : !firrtl.uint<1>
    firrtl.strictconnect %c, %a : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %b, [[CONST]]
    firrtl.strictconnect %b, %c : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module private @blockProp3
  firrtl.module private @blockProp3(in %clock: !firrtl.clock, in %a: !firrtl.uint<1> , out %b: !firrtl.uint<1>) {
    //CHECK: %c = firrtl.reg
    %c = firrtl.reg sym @s2 %clock : !firrtl.clock, !firrtl.uint<1>
    firrtl.strictconnect %c, %a : !firrtl.uint<1>
    firrtl.strictconnect %b, %c : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @testDontTouch
  firrtl.module @testDontTouch(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>, out %a1: !firrtl.uint<1>, out %a2: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %blockProp1_clock, %blockProp1_a, %blockProp1_b = firrtl.instance blockProp1 sym @a1 @blockProp1(in clock: !firrtl.clock, in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %allowProp_clock, %allowProp_a, %allowProp_b = firrtl.instance allowProp sym @a2 @allowProp(in clock: !firrtl.clock, in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %blockProp3_clock, %blockProp3_a, %blockProp3_b = firrtl.instance blockProp3  sym @a3 @blockProp3(in clock: !firrtl.clock, in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    firrtl.strictconnect %blockProp1_clock, %clock : !firrtl.clock
    firrtl.strictconnect %allowProp_clock, %clock : !firrtl.clock
    firrtl.strictconnect %blockProp3_clock, %clock : !firrtl.clock
    firrtl.strictconnect %blockProp1_a, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %allowProp_a, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %blockProp3_a, %c1_ui1 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %a, %blockProp1_b
    firrtl.strictconnect %a, %blockProp1_b : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %a1, %c
    firrtl.strictconnect %a1, %allowProp_b : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %a2, %blockProp3_b
    firrtl.strictconnect %a2, %blockProp3_b : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @CheckNode
  firrtl.module @CheckNode(out %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK-NOT: %d1 = firrtl.node
    %d1 = firrtl.node droppable_name %c1_ui1 : !firrtl.uint<1>
    // CHECK: %d2 = firrtl.node
    %d2 = firrtl.node interesting_name %c1_ui1 : !firrtl.uint<1>
    // CHECK: %d3 = firrtl.node
    %d3 = firrtl.node   sym @s2 %c1_ui1: !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %x, %c1_ui1
    firrtl.strictconnect %x, %d1 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %y, %c1_ui1
    firrtl.strictconnect %y, %d2 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %z, %d3
    firrtl.strictconnect %z, %d3 : !firrtl.uint<1>
  }

}

// -----

firrtl.circuit "OutPortTop" {
    firrtl.module private @OutPortChild1(out %out: !firrtl.uint<1> sym @dntSym1) {
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      firrtl.strictconnect %out, %c0_ui1 : !firrtl.uint<1>
    }
    firrtl.module private @OutPortChild2(out %out: !firrtl.uint<1>) {
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      firrtl.strictconnect %out, %c0_ui1 : !firrtl.uint<1>
    }
  // CHECK-LABEL: firrtl.module @OutPortTop
    firrtl.module @OutPortTop(in %x: !firrtl.uint<1>, out %zc: !firrtl.uint<1>, out %zn: !firrtl.uint<1>) {
      // CHECK: %c0_ui1 = firrtl.constant 0
      %c_out = firrtl.instance c  sym @a2 @OutPortChild1(out out: !firrtl.uint<1>)
      %c_out_0 = firrtl.instance c  sym @a1 @OutPortChild2(out out: !firrtl.uint<1>)
      // CHECK: %0 = firrtl.and %x, %c_out
      %0 = firrtl.and %x, %c_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      %1 = firrtl.and %x, %c_out_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK: firrtl.strictconnect %zn, %0
      firrtl.strictconnect %zn, %0 : !firrtl.uint<1>
      // CHECK: firrtl.strictconnect %zc, %c0_ui1
      firrtl.strictconnect %zc, %1 : !firrtl.uint<1>
    }
}


// -----

firrtl.circuit "InputPortTop"   {
  // CHECK-LABEL: firrtl.module private @InputPortChild2
  firrtl.module private @InputPortChild2(in %in0: !firrtl.uint<1>, in %in1: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    // CHECK: = firrtl.constant 1
    %0 = firrtl.and %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %out, %0 : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module private @InputPortChild
  firrtl.module private @InputPortChild(in %in0: !firrtl.uint<1>,
    in %in1 : !firrtl.uint<1> sym @dntSym1, out %out: !firrtl.uint<1>) {
    // CHECK: %0 = firrtl.and %in0, %in1
    %0 = firrtl.and %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %out, %0 : !firrtl.uint<1>
  }
  firrtl.module @InputPortTop(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>, out %z2: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c_in0, %c_in1, %c_out = firrtl.instance c @InputPortChild(in in0: !firrtl.uint<1>, in in1: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %c2_in0, %c2_in1, %c2_out = firrtl.instance c2 @InputPortChild2(in in0: !firrtl.uint<1>, in in1: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %z, %c_out : !firrtl.uint<1>
    firrtl.strictconnect %c_in0, %x : !firrtl.uint<1>
    firrtl.strictconnect %c_in1, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %z2, %c2_out : !firrtl.uint<1>
    firrtl.strictconnect %c2_in0, %x : !firrtl.uint<1>
    firrtl.strictconnect %c2_in1, %c1_ui1 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "InstanceOut"   {
  firrtl.extmodule private @Ext(in a: !firrtl.uint<1>)

  // CHECK-LABEL: firrtl.module @InstanceOut
  firrtl.module @InstanceOut(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %ext_a = firrtl.instance ext @Ext(in a: !firrtl.uint<1>)
    firrtl.strictconnect %ext_a, %a : !firrtl.uint<1>
    %w = firrtl.wire  : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %w, %ext_a : !firrtl.uint<1>
    firrtl.strictconnect %w, %ext_a : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %b, %w : !firrtl.uint<1>
    firrtl.strictconnect %b, %w : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "InstanceOut2"   {
  firrtl.module private @Ext(in %a: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: firrtl.module @InstanceOut2
  firrtl.module @InstanceOut2(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %ext_a = firrtl.instance ext @Ext(in a: !firrtl.uint<1>)
    firrtl.strictconnect %ext_a, %a : !firrtl.uint<1>
    %w = firrtl.wire  : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %w, %ext_a : !firrtl.uint<1>
    firrtl.strictconnect %w, %ext_a : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %b, %w : !firrtl.uint<1>
    firrtl.strictconnect %b, %w : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "invalidReg1"   {
  // CHECK-LABEL: @invalidReg1
  firrtl.module @invalidReg1(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %foobar = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
      //CHECK: %0 = firrtl.not %foobar : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %0 = firrtl.not %foobar : (!firrtl.uint<1>) -> !firrtl.uint<1>
      //CHECK: firrtl.strictconnect %foobar, %0 : !firrtl.uint<1>
      firrtl.strictconnect %foobar, %0 : !firrtl.uint<1>
      //CHECK: firrtl.strictconnect %a, %foobar : !firrtl.uint<1>
      firrtl.strictconnect %a, %foobar : !firrtl.uint<1>
  }
}

// -----

// This test is checking the behavior of a RegOp, "r", and a RegResetOp, "s",
// that are combinationally connected to themselves through simple and weird
// formulations.  In all cases it should NOT be optimized away.  For more discussion, see:
//   - https://github.com/llvm/circt/issues/1465
//   - https://github.com/llvm/circt/issues/1466
//   - https://github.com/llvm/circt/issues/1478
//
// CHECK-LABEL: "Oscillators"
firrtl.circuit "Oscillators"   {
  // CHECK: firrtl.module private @Foo
  firrtl.module private @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.uint<1>) {
    // CHECK: firrtl.reg
    %r = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: firrtl.regreset
    %s = firrtl.regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %0 = firrtl.not %r : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %r, %0 : !firrtl.uint<1>
    %1 = firrtl.not %s : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %s, %1 : !firrtl.uint<1>
    %2 = firrtl.or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %a, %2 : !firrtl.uint<1>
  }
  // CHECK: firrtl.module private @Bar
  firrtl.module private @Bar(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.uint<1>) {
    // CHECK: %r = firrtl.reg
    %r = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: firrtl.regreset
    %s = firrtl.regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.xor %a, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %r, %0 : !firrtl.uint<1>
    %1 = firrtl.xor %a, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %s, %1 : !firrtl.uint<1>
    %2 = firrtl.or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %a, %2 : !firrtl.uint<1>
  }
  // CHECK: firrtl.module private @Baz
  firrtl.module private @Baz(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.uint<1>) {
    // CHECK: firrtl.reg
    %r = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: firrtl.regreset
    %s = firrtl.regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %0 = firrtl.not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %r, %0 : !firrtl.uint<1>
    %1 = firrtl.not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %s, %1 : !firrtl.uint<1>
    %2 = firrtl.or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %a, %2 : !firrtl.uint<1>
  }
  firrtl.extmodule @Ext(in a: !firrtl.uint<1>)
  // CHECK: firrtl.module private @Qux
  firrtl.module private @Qux(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.uint<1>) {
    %ext_a = firrtl.instance ext @Ext(in a: !firrtl.uint<1>)
    // CHECK: firrtl.reg
    %r = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: firrtl.regreset
    %s = firrtl.regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %0 = firrtl.not %ext_a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %r, %0 : !firrtl.uint<1>
    %1 = firrtl.not %ext_a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %s, %1 : !firrtl.uint<1>
    %2 = firrtl.or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %ext_a, %2 : !firrtl.uint<1>
    firrtl.strictconnect %a, %ext_a : !firrtl.uint<1>
  }
  firrtl.module @Oscillators(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %foo_a: !firrtl.uint<1>, out %bar_a: !firrtl.uint<1>, out %baz_a: !firrtl.uint<1>, out %qux_a: !firrtl.uint<1>) {
    %foo_clock, %foo_reset, %foo_a_0 = firrtl.instance foo @Foo(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.uint<1>)
    firrtl.strictconnect %foo_clock, %clock : !firrtl.clock
    firrtl.strictconnect %foo_reset, %reset : !firrtl.asyncreset
    firrtl.strictconnect %foo_a, %foo_a_0 : !firrtl.uint<1>
    %bar_clock, %bar_reset, %bar_a_1 = firrtl.instance bar @Bar (in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.uint<1>)
    firrtl.strictconnect %bar_clock, %clock : !firrtl.clock
    firrtl.strictconnect %bar_reset, %reset : !firrtl.asyncreset
    firrtl.strictconnect %bar_a, %bar_a_1 : !firrtl.uint<1>
    %baz_clock, %baz_reset, %baz_a_2 = firrtl.instance baz @Baz(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.uint<1>)
    firrtl.strictconnect %baz_clock, %clock : !firrtl.clock
    firrtl.strictconnect %baz_reset, %reset : !firrtl.asyncreset
    firrtl.strictconnect %baz_a, %baz_a_2 : !firrtl.uint<1>
    %qux_clock, %qux_reset, %qux_a_3 = firrtl.instance qux @Qux(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.uint<1>)
    firrtl.strictconnect %qux_clock, %clock : !firrtl.clock
    firrtl.strictconnect %qux_reset, %reset : !firrtl.asyncreset
    firrtl.strictconnect %qux_a, %qux_a_3 : !firrtl.uint<1>
  }
}

// -----

// This test checks that an output port sink, used as a RHS of a connect, is not
// optimized away.  This is similar to the oscillator tests above, but more
// reduced. See:
//   - https://github.com/llvm/circt/issues/1488
//
// CHECK-LABEL: firrtl.circuit "rhs_sink_output_used_as_wire"
firrtl.circuit "rhs_sink_output_used_as_wire" {
  // CHECK: firrtl.module private @Bar
  firrtl.module private @Bar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    firrtl.strictconnect %c, %b : !firrtl.uint<1>
    %_c = firrtl.wire  : !firrtl.uint<1>
    // CHECK: firrtl.xor %a, %c
    %0 = firrtl.xor %a, %c : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %_c, %0 : !firrtl.uint<1>
    firrtl.strictconnect %d, %_c : !firrtl.uint<1>
  }
  firrtl.module @rhs_sink_output_used_as_wire(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %bar_a, %bar_b, %bar_c, %bar_d = firrtl.instance bar @Bar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d: !firrtl.uint<1>)
    firrtl.strictconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.strictconnect %bar_b, %b : !firrtl.uint<1>
    firrtl.strictconnect %c, %bar_c : !firrtl.uint<1>
    firrtl.strictconnect %d, %bar_d : !firrtl.uint<1>
  }
}

// -----

// issue 1793
// Ensure don't touch on output port is seen by instances
firrtl.circuit "dntOutput" {
  // CHECK-LABEL: firrtl.module @dntOutput
  // CHECK: %0 = firrtl.mux(%c, %int_b, %c2_ui3)
  // CHECK-NEXT: firrtl.strictconnect %b, %0
  firrtl.module @dntOutput(out %b : !firrtl.uint<3>, in %c : !firrtl.uint<1>) {
    %const = firrtl.constant 2 : !firrtl.uint<3>
    %int_b = firrtl.instance int @foo(out b: !firrtl.uint<3>)
    %m = firrtl.mux(%c, %int_b, %const) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.strictconnect %b, %m : !firrtl.uint<3>
  }
  firrtl.module private @foo(out %b: !firrtl.uint<3>  sym @dntSym1) {
    %const = firrtl.constant 1 : !firrtl.uint<3>
    firrtl.strictconnect %b, %const : !firrtl.uint<3>
  }
}

// -----

// An annotation should block removal of a wire, but should not block constant
// folding.
//
// CHECK-LABEL: "AnnotationsBlockRemoval"
firrtl.circuit "AnnotationsBlockRemoval"  {
  firrtl.module @AnnotationsBlockRemoval(out %b: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: %w = firrtl.wire
    %w = firrtl.wire droppable_name {annotations = [{class = "foo"}]} : !firrtl.uint<1>
    firrtl.strictconnect %w, %c1_ui1 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %b, %c1_ui1
    firrtl.strictconnect %b, %w : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: "Issue3372"
firrtl.circuit "Issue3372"  {
  firrtl.module @Issue3372(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %value: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %other_zero = firrtl.instance other interesting_name  @Other(out zero: !firrtl.uint<1>)
    %shared = firrtl.regreset interesting_name %clock, %other_zero, %c1_ui1  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.strictconnect %shared, %shared : !firrtl.uint<1>
    %test = firrtl.wire interesting_name  : !firrtl.uint<1>
    firrtl.strictconnect %test, %shared : !firrtl.uint<1>
    firrtl.strictconnect %value, %test : !firrtl.uint<1>
  }
// CHECK:  %other_zero = firrtl.instance other interesting_name @Other(out zero: !firrtl.uint<1>)
// CHECK:  %test = firrtl.wire interesting_name : !firrtl.uint<1>
// CHECK:  firrtl.strictconnect %value, %test : !firrtl.uint<1>

  firrtl.module private @Other(out %zero: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.strictconnect %zero, %c0_ui1 : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: "SendThroughRef"
firrtl.circuit "SendThroughRef" {
  firrtl.module private @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %ref_zero = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %ref_zero : !firrtl.probe<uint<1>>
  }
  // CHECK:  firrtl.strictconnect %a, %c0_ui1 : !firrtl.uint<1>
  firrtl.module @SendThroughRef(out %a: !firrtl.uint<1>) {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.probe<uint<1>>)
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: "ForwardRef"
firrtl.circuit "ForwardRef" {
  firrtl.module private @RefForward2(out %_a: !firrtl.probe<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %ref_zero = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %ref_zero : !firrtl.probe<uint<1>>
  }
  firrtl.module private @RefForward(out %_a: !firrtl.probe<uint<1>>) {
    %fwd_2 = firrtl.instance fwd_2 @RefForward2(out _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %_a, %fwd_2 : !firrtl.probe<uint<1>>
  }
  // CHECK:  firrtl.strictconnect %a, %c0_ui1 : !firrtl.uint<1>
  firrtl.module @ForwardRef(out %a: !firrtl.uint<1>) {
    %fwd_a = firrtl.instance fwd @RefForward(out _a: !firrtl.probe<uint<1>>)
    %0 = firrtl.ref.resolve %fwd_a : !firrtl.probe<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Don't prop through a rwprobe ref.

// CHECK-LABEL: "SendThroughRWProbe"
firrtl.circuit "SendThroughRWProbe" {
  // CHECK-LABEL: firrtl.module private @Bar
  firrtl.module private @Bar(out %rw: !firrtl.rwprobe<uint<1>>, out %out : !firrtl.uint<1>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[N:.+]], %{{.+}} = firrtl.node
    // CHECK-SAME: forceable
    %n, %n_ref = firrtl.node %zero forceable : !firrtl.uint<1>
    // CHECK: firrtl.node %[[N]]
    %user = firrtl.node %n : !firrtl.uint<1>
    firrtl.strictconnect %out, %user : !firrtl.uint<1>
    firrtl.ref.define %rw, %n_ref : !firrtl.rwprobe<uint<1>>
  }
  // CHECK:  firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  firrtl.module @SendThroughRWProbe(out %a: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %bar_rw, %bar_out = firrtl.instance bar @Bar(out rw: !firrtl.rwprobe<uint<1>>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %out, %bar_out : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_rw : !firrtl.rwprobe<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Should not crash when there are properties.

// CHECK-LABEL: firrtl.circuit "Properties"
firrtl.circuit "Properties" {
  firrtl.module @Properties(in %in : !firrtl.string, out %out : !firrtl.string) {
    firrtl.propassign %out, %in : !firrtl.string
  }
}

// -----

// Verbatim expressions should not be optimized away.
firrtl.circuit "Verbatim"  {
  firrtl.module @Verbatim() {
    // CHECK: %[[v0:.+]] = firrtl.verbatim.expr
    %0 = firrtl.verbatim.expr "random.something" : () -> !firrtl.uint<1>
    // CHECK: %tap = firrtl.wire   : !firrtl.uint<1>
    %tap = firrtl.wire   : !firrtl.uint<1>
    %fizz = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    firrtl.strictconnect %fizz, %tap : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %tap, %[[v0]] : !firrtl.uint<1>
    firrtl.strictconnect %tap, %0 : !firrtl.uint<1>
    // CHECK: firrtl.verbatim.wire "randomBar.b"
    %1 = firrtl.verbatim.wire "randomBar.b" : () -> !firrtl.uint<1> {symbols = []}
    // CHECK: %tap2 = firrtl.wire   : !firrtl.uint<1>
    %tap2 = firrtl.wire   : !firrtl.uint<1>
    firrtl.strictconnect %tap2, %1 : !firrtl.uint<1>
  }
}

// -----

// This test is only checking that IMCP doesn't generate invalid IR.  IMCP needs
// to delete the strictconnect instead of replacing its destination with an
// invalid value that will replace the register.  For more information, see:
//   - https://github.com/llvm/circt/issues/4498
//
// CHECK-LABEL: "Issue4498"
firrtl.circuit "Issue4498"  {
  firrtl.module @Issue4498(in %clock: !firrtl.clock) {
    %a = firrtl.wire : !firrtl.uint<1>
    %r = firrtl.reg interesting_name %clock : !firrtl.clock, !firrtl.uint<1>
    firrtl.strictconnect %r, %a : !firrtl.uint<1>
  }
}

// -----

// An ordering dependnecy crept in with unwritten.  Check that it's gone
// CHECK-LABEL: "Ordering"
firrtl.circuit "Ordering" {
  firrtl.module public @Ordering(out %b: !firrtl.uint<1>) {
    %0 = firrtl.wire : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.strictconnect %0, %c1_ui1 : !firrtl.uint<1>
    %1 = firrtl.xor %0, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %b, %1 : !firrtl.uint<1>
  }
  // CHECK: firrtl.strictconnect %b, %c0_ui1
}

// -----

// Checking that plusargs intrinsics are properly marked overdefined in IMCP,
// see:
//   - https://github.com/llvm/circt/issues/5722
//
// CHECK-LABEL: "Issue5722Test"
firrtl.circuit "Issue5722Test"  {
  firrtl.module @Issue5722Test(out %a: !firrtl.uint<1>) {
    %b = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.int.plusargs.test "parg"
    firrtl.strictconnect %b, %0 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %b, %0
    firrtl.strictconnect %a, %b : !firrtl.uint<1>
  }
}

// CHECK-LABEL: "Issue5722Value"
firrtl.circuit "Issue5722Value"  {
  firrtl.module @Issue5722Value(out %a: !firrtl.uint<1>, out %v: !firrtl.uint<32>) {
    %b = firrtl.wire : !firrtl.uint<1>
    %c = firrtl.wire : !firrtl.uint<32>
    %0:2 = firrtl.int.plusargs.value "parg" : !firrtl.uint<32>
    firrtl.strictconnect %b, %0#0 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %b, %0#0
    firrtl.strictconnect %c, %0#1 : !firrtl.uint<32>
    // CHECK: firrtl.strictconnect %c, %0#1
    firrtl.strictconnect %a, %b : !firrtl.uint<1>
    firrtl.strictconnect %v, %c : !firrtl.uint<32>
  }
}

// -----

// Check const prop of basic properties.

// CHECK-LABEL: "PropPassthruTest"
firrtl.circuit "PropPassthruTest" {
  firrtl.module private @Passthru(in %intIn: !firrtl.integer,
                                  out %intOut: !firrtl.integer,
                                  in %boolIn: !firrtl.bool,
                                  out %boolOut: !firrtl.bool,
                                  in %strIn: !firrtl.string,
                                  out %strOut: !firrtl.string) {
    firrtl.propassign %intOut, %intIn : !firrtl.integer
    firrtl.propassign %boolOut, %boolIn : !firrtl.bool
    firrtl.propassign %strOut, %strIn : !firrtl.string
  }
  firrtl.module @PropPassthruTest(out %intOut: !firrtl.integer,
                                  out %boolOut: !firrtl.bool,
                                  out %strOut: !firrtl.string) {
    // CHECK-DAG: %[[BOOL:.+]] = firrtl.bool true
    // CHECK-DAG: %[[STRING:.+]] = firrtl.string "hello"
    // CHECK-DAG: %[[INT:.+]] = firrtl.integer 123
    %0 = firrtl.bool true
    %1 = firrtl.string "hello"
    %2 = firrtl.integer 123
    %passthru_intIn, %passthru_intOut, %passthru_boolIn, %passthru_boolOut, %passthru_strIn, %passthru_strOut = firrtl.instance passthru @Passthru(in intIn: !firrtl.integer, out intOut: !firrtl.integer, in boolIn: !firrtl.bool, out boolOut: !firrtl.bool, in strIn: !firrtl.string, out strOut: !firrtl.string)
    firrtl.propassign %passthru_intIn, %2 : !firrtl.integer
    firrtl.propassign %passthru_strIn, %1 : !firrtl.string
    firrtl.propassign %passthru_boolIn, %0 : !firrtl.bool
    // CHECK-DAG: propassign %intOut, %[[INT]]
    // CHECK-DAG: propassign %strOut, %[[STRING]]
    // CHECK-DAG: propassign %boolOut, %[[BOOL]]
    firrtl.propassign %intOut, %passthru_intOut : !firrtl.integer
    firrtl.propassign %strOut, %passthru_strOut : !firrtl.string
    firrtl.propassign %boolOut, %passthru_boolOut : !firrtl.bool
  }
}

// -----
// Check assignments into object fields don't cause errors.

// CHECK-LABEL: "ObjectSubfieldConnect"
firrtl.circuit "ObjectSubfieldConnect" {
  firrtl.class private @Test(in %in: !firrtl.integer) {
  }
  firrtl.module @ObjectSubfieldConnect(in %in: !firrtl.integer) {
    %0 = firrtl.object @Test(in in: !firrtl.integer)
    %1 = firrtl.object.subfield %0[in] : !firrtl.class<@Test(in in: !firrtl.integer)>
    firrtl.propassign %1, %in : !firrtl.integer
  }
}

// -----
// Check assignments of objects themselves are handled.

// CHECK-LABEL: "ObjectConnect"
firrtl.circuit "ObjectConnect" {
  firrtl.class private @Test() {}
  firrtl.module @Passthru(in %in : !firrtl.class<@Test()>,
                          out %out : !firrtl.class<@Test()>) {
    firrtl.propassign %out, %in : !firrtl.class<@Test()>
  }
  firrtl.module @ObjectConnect(out %out : !firrtl.class<@Test()>) {
    %c_in, %c_out = firrtl.instance c @Passthru(in in : !firrtl.class<@Test()>,
                                                out out : !firrtl.class<@Test()>)
    %obj = firrtl.object @Test()
    firrtl.propassign %c_in, %obj : !firrtl.class<@Test()>
    firrtl.propassign %out, %c_out : !firrtl.class<@Test()>
  }
}

// -----

// Preserve forceable decl that is dead other than its rwprobe result.
// CHECK-LABEL: "KeepForceable"
firrtl.circuit "KeepForceable" {
  firrtl.module @KeepForceable(out %a: !firrtl.rwprobe<uint<1>>) attributes {convention = #firrtl<convention scalarized>} {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %b_c = firrtl.wire : !firrtl.rwprobe<uint<1>>
    %d, %d_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    firrtl.strictconnect %d, %c0_ui1 : !firrtl.uint<1>
    // CHECK-COUNT-2: ref.define
    firrtl.ref.define %b_c, %d_ref : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %a, %b_c : !firrtl.rwprobe<uint<1>>
  }
}

// -----

// Fix partly deleting non-hw dataflow.
// Issue #6076.

// CHECK-LABEL: "WireProbe"
firrtl.circuit "WireProbe" {
  // CHECK: module @WireProbe
  firrtl.module @WireProbe(out %p : !firrtl.probe<uint<5>>) {
    // CHECK-NEXT: %[[ZERO:.+]] = firrtl.constant 0
    // CHECK-NEXT: %[[REF:.+]] = firrtl.ref.send %[[ZERO]]
    // CHECK-NEXT: firrtl.ref.define %p, %[[REF]]
    // CHECK-NEXT: }
    %x = firrtl.constant 0: !firrtl.uint<5>
    %0 = firrtl.ref.send %x : !firrtl.uint<5>
    %w = firrtl.wire interesting_name : !firrtl.probe<uint<5>>
    firrtl.ref.define %w, %0: !firrtl.probe<uint<5>>
    firrtl.ref.define %p, %w : !firrtl.probe<uint<5>>
  }
}

// -----

// Also for properties that we const-prop.
// Drop connections to, and wires, that are not overdefined.

// CHECK-LABEL: firrtl.circuit "WireProp"
firrtl.circuit "WireProp" {
  // CHECK-NOT: firrtl.wire
  firrtl.module @WireProp(out %s : !firrtl.string) {
    %x = firrtl.string "hello"
    %w = firrtl.wire : !firrtl.string
    %w2 = firrtl.wire : !firrtl.string
    firrtl.propassign %w, %x: !firrtl.string
    firrtl.propassign %w2, %w: !firrtl.string
    firrtl.propassign %s, %w : !firrtl.string
  }
}

// -----

// Check ability to const-prop through declarations (wires, nodes)
// with annotations, but keep them around.
firrtl.circuit "ConstPropAnno" {
  firrtl.module @ConstPropAnno(out %val : !firrtl.uint<3>,
                               out %val2 : !firrtl.uint<3>) {
    // CHECK: %[[ZERO:.+]] = firrtl.constant 0
    %zero = firrtl.constant 0 : !firrtl.uint<3>
    // CHECK: %w = firrtl.wire
    %w = firrtl.wire {annotations = [{class = "circt.test"}]} : !firrtl.uint<3>
    // CHECK-NOT: firrtl.wire
    %w2 = firrtl.wire : !firrtl.uint<3>
    firrtl.strictconnect %w2, %zero : !firrtl.uint<3>
    firrtl.strictconnect %w, %w2 : !firrtl.uint<3>
    firrtl.strictconnect %val, %w : !firrtl.uint<3>

    // CHECK: firrtl.node %[[ZERO]]
    %n = firrtl.node %w2 {annotations = [{class = "circt.test"}]} : !firrtl.uint<3>
    // CHECK-NOT: firrtl.wire
    %w3 = firrtl.wire : !firrtl.uint<3>
    firrtl.strictconnect %w3, %n : !firrtl.uint<3>
    firrtl.strictconnect %val2, %w3 : !firrtl.uint<3>
  }
}

// -----

// RefSubOp: preserve.

// CHECK-LABEL: "RefSubOp"
firrtl.circuit "RefSubOp" {
  firrtl.extmodule private @Ext(out p : !firrtl.probe<vector<uint<32>, 31>>)
  firrtl.module @RefSubOp(out %p : !firrtl.probe<uint<32>>) {
   // CHECK: firrtl.wire
   // CHECK: ref.sub
   // CHECK-COUNT-2: firrtl.ref.define
   %ext_vec_ref = firrtl.instance e @Ext(out p : !firrtl.probe<vector<uint<32>, 31>>)
   %w = firrtl.wire { name = "tap" } : !firrtl.probe<uint<32>>
   %ref = firrtl.ref.sub %ext_vec_ref[5] : !firrtl.probe<vector<uint<32>, 31>>
   firrtl.ref.define %w, %ref : !firrtl.probe<uint<32>>
   firrtl.ref.define %p, %w: !firrtl.probe<uint<32>>
  }
}

// -----

// RefSubOp: prop.

// CHECK-LABEL: "RefSubOpPropagate"
firrtl.circuit "RefSubOpPropagate" {
  firrtl.module private @Child(out %p : !firrtl.probe<vector<uint<32>, 2>>,
                               out %w0 : !firrtl.probe<uint<32>>,
                               out %w1 : !firrtl.probe<uint<32>>) {
    // Vector initialized to 123, 321
    %w = firrtl.wire : !firrtl.vector<uint<32>, 2>
    %v0 = firrtl.subindex %w[0] : !firrtl.vector<uint<32>, 2>
    %v1 = firrtl.subindex %w[1] : !firrtl.vector<uint<32>, 2>
    %val0 = firrtl.constant 123 : !firrtl.uint<32>
    %val1 = firrtl.constant 321 : !firrtl.uint<32>
    firrtl.strictconnect %v0, %val0 : !firrtl.uint<32>
    firrtl.strictconnect %v1, %val1 : !firrtl.uint<32>

    // Send out probe of entire vector.
    %vec_ref = firrtl.ref.send %w : !firrtl.vector<uint<32>, 2>
    firrtl.ref.define %p, %vec_ref : !firrtl.probe<vector<uint<32>, 2>>

    // Send out probe of individual elements.
    %w0_ref = firrtl.ref.sub %vec_ref[0] : !firrtl.probe<vector<uint<32>, 2>>
    %w1_ref = firrtl.ref.sub %vec_ref[1] : !firrtl.probe<vector<uint<32>, 2>>
    firrtl.ref.define %w0, %w0_ref : !firrtl.probe<uint<32>>
    firrtl.ref.define %w1, %w1_ref : !firrtl.probe<uint<32>>
  }

   // CHECK: module @RefSubOpPropagate
  firrtl.module @RefSubOpPropagate(out %p : !firrtl.probe<uint<32>>,
                                   out %w1_via_p : !firrtl.uint<32>,
                                   out %w0 : !firrtl.uint<32>,
                                   out %w1 : !firrtl.uint<32>) {
   // Constant pool
   // CHECK-DAG: %[[C123:.+]] = firrtl.constant 123
   // CHECK-DAG: %[[C321:.+]] = firrtl.constant 321
   // CHECK-DAG: %[[P321:.+]] = firrtl.ref.send %[[C321]]
   // CHECK: firrtl.instance
   %c_vec_ref, %c_w0, %c_w1 = firrtl.instance c @Child(out p : !firrtl.probe<vector<uint<32>, 2>>,
                                                       out w0 : !firrtl.probe<uint<32>>,
                                                       out w1 : !firrtl.probe<uint<32>>)

   // 'p' should be probe of second element.
   // CHECK-NEXT: firrtl.ref.define %p, %[[P321]]
   %w = firrtl.wire : !firrtl.probe<uint<32>>
   %ref = firrtl.ref.sub %c_vec_ref[1] : !firrtl.probe<vector<uint<32>, 2>>
   firrtl.ref.define %w, %ref : !firrtl.probe<uint<32>>
   firrtl.ref.define %p, %w: !firrtl.probe<uint<32>>

   // CHECK-NEXT: firrtl.strictconnect %w1_via_p, %[[C321]]
   %p_read = firrtl.ref.resolve %ref : !firrtl.probe<uint<32>>
   firrtl.strictconnect %w1_via_p, %p_read : !firrtl.uint<32>

   // CHECK-NEXT: firrtl.strictconnect %w0, %[[C123]]
   // CHECK-NEXT: firrtl.strictconnect %w1, %[[C321]]
   %w0_read = firrtl.ref.resolve %c_w0 : !firrtl.probe<uint<32>>
   %w1_read = firrtl.ref.resolve %c_w1 : !firrtl.probe<uint<32>>
   firrtl.strictconnect %w0, %w0_read : !firrtl.uint<32>
   firrtl.strictconnect %w1, %w1_read : !firrtl.uint<32>
   // CHECK-NEXT: }
  }
}

// -----

// OMIR annotation should not block removal.
//   - See: https://github.com/llvm/circt/issues/6199
//
// CHECK-LABEL: firrtl.circuit "OMIRRemoval"
firrtl.circuit "OMIRRemoval" {
  firrtl.module @OMIRRemoval(
    out %a: !firrtl.uint<1>,
    out %b: !firrtl.uint<2>,
    out %c: !firrtl.uint<3>,
    out %d: !firrtl.uint<4>
  ) {
    %c1_ui1  = firrtl.constant 1  : !firrtl.uint<1>
    %c3_ui2  = firrtl.constant 3  : !firrtl.uint<2>
    %c7_ui3  = firrtl.constant 7  : !firrtl.uint<3>
    %c15_ui4 = firrtl.constant 15 : !firrtl.uint<4>

    // CHECK-NOT: %tmp_0
    %tmp_0 = firrtl.node %c1_ui1 {
      annotations = [
        {
           class = "freechips.rocketchip.objectmodel.OMIRTracker",
           id = 0 : i64,
           type = "OMReferenceTarget"
        }
      ]} : !firrtl.uint<1>
    firrtl.strictconnect %a, %tmp_0 : !firrtl.uint<1>

    // CHECK-NOT: %tmp_1
    %tmp_1 = firrtl.node %c3_ui2 {
      annotations = [
        {
           class = "freechips.rocketchip.objectmodel.OMIRTracker",
           id = 1 : i64,
           type = "OMMemberReferenceTarget"
        }
      ]} : !firrtl.uint<2>
    firrtl.strictconnect %b, %tmp_1 : !firrtl.uint<2>

    // CHECK-NOT: %tmp_2
    %tmp_2 = firrtl.node %c7_ui3 {
      annotations = [
        {
           class = "freechips.rocketchip.objectmodel.OMIRTracker",
           id = 3 : i64,
           type = "OMMemberInstanceTarget"
        }
      ]} : !firrtl.uint<3>
    firrtl.strictconnect %c, %tmp_2 : !firrtl.uint<3>

    // Adding one additional annotation will block removal.
    //
    // CHECK: %tmp_3
    %tmp_3 = firrtl.node %c15_ui4 {
      annotations = [
        {
           class = "freechips.rocketchip.objectmodel.OMIRTracker",
           id = 4 : i64,
           type = "OMReferenceTarget"
        },
        {
           class = "circt.test"
        }
      ]} : !firrtl.uint<4>
    // CHECK-NEXT: firrtl.strictconnect %d, %c15_ui4
    firrtl.strictconnect %d, %tmp_3 : !firrtl.uint<4>
  }
}
