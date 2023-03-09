// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop), canonicalize{top-down region-simplify}, firrtl.circuit(firrtl.module(firrtl-register-optimizer)))'  %s | FileCheck %s
// github.com/chipsalliance/firrtl: test/scala/firrtlTests/ConstantPropagationTests.scala

//propagate constant inputs  
firrtl.circuit "ConstInput"   {
  firrtl.module @ConstInput(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c_in0, %c_in1, %c_out = firrtl.instance c @Child(in in0: !firrtl.uint<1>, in in1: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %c_in0, %x : !firrtl.uint<1>
    firrtl.strictconnect %c_in1, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %z, %c_out : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module private @Child
  firrtl.module private @Child(in %in0: !firrtl.uint<1>, in %in1: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %0 = firrtl.and %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %out, %in0 :
    firrtl.strictconnect %out, %0 : !firrtl.uint<1>
  }
}

//propagate constant inputs ONLY if ALL instance inputs get the same value
firrtl.circuit "InstanceInput"   {
  // CHECK-LABEL: firrtl.module private @Bottom1
  firrtl.module private @Bottom1(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
      // CHECK: %c1_ui1 = firrtl.constant 1
      // CHECK: firrtl.strictconnect %out, %c1_ui1
    firrtl.strictconnect %out, %in : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module private @Child1
  firrtl.module private @Child1(out %out: !firrtl.uint<1>) {
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    %b0_in, %b0_out = firrtl.instance b0 @Bottom1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %b0_in, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    // CHECK: %[[C1:.+]] = firrtl.constant 1 :
    // CHECK: firrtl.strictconnect %out, %[[C1]]
    firrtl.strictconnect %out, %b0_out : !firrtl.uint<1>
  }
  // CHECK-LABEL:  firrtl.module @InstanceInput
  firrtl.module @InstanceInput(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    %c_out = firrtl.instance c @Child1(out out: !firrtl.uint<1>)
    %b0_in, %b0_out = firrtl.instance b0  @Bottom1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %b0_in, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %b1_in, %b1_out = firrtl.instance b1  @Bottom1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %b1_in, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %0 = firrtl.and %b0_out, %b1_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %1 = firrtl.and %0, %c_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: %[[C0:.+]] = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %z, %[[C0]] : !firrtl.uint<1>
    firrtl.strictconnect %z, %1 : !firrtl.uint<1>
  }
}

//propagate constant inputs ONLY if ALL instance inputs get the same value
firrtl.circuit "InstanceInput2"   {
  // CHECK-LABEL: firrtl.module private @Bottom2
  firrtl.module private @Bottom2(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    // CHECK: firrtl.strictconnect %out, %in 
    firrtl.strictconnect %out, %in : !firrtl.uint<1>
  }
 // CHECK-LABEL:  firrtl.module private @Child2
  firrtl.module private @Child2(out %out: !firrtl.uint<1>) {
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    %b0_in, %b0_out = firrtl.instance b0 @Bottom2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %b0_in, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    // CHECK: firrtl.strictconnect %out, %b0_out
    firrtl.strictconnect %out, %b0_out : !firrtl.uint<1>
  }
 // CHECK-LABEL:  firrtl.module @InstanceInput2
  firrtl.module @InstanceInput2(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    %c_out = firrtl.instance c @Child2(out out: !firrtl.uint<1>)
    %b0_in, %b0_out = firrtl.instance b0 @Bottom2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %b0_in, %x : !firrtl.uint<1>
    %b1_in, %b1_out = firrtl.instance b1 @Bottom2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %b1_in, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %0 = firrtl.and %b0_out, %b1_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %1 = firrtl.and %0, %c_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
   // CHECK:  firrtl.strictconnect %z, %1
    firrtl.strictconnect %z, %1 : !firrtl.uint<1>
  }
}

// ConstProp should work across wires
firrtl.circuit "acrossWire"   {
  // CHECK-LABEL: firrtl.module @acrossWire
  firrtl.module @acrossWire(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    %_z = firrtl.wire droppable_name : !firrtl.uint<1>
    firrtl.strictconnect %y, %_z : !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.mux(%x, %c0_ui1, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %_z, %0 : !firrtl.uint<1>
    // CHECK: %[[C2:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %y, %[[C2]] : !firrtl.uint<1>
  }
}

//"ConstProp" should "propagate constant outputs"
firrtl.circuit "constOutput"   {
  firrtl.module private @constOutChild(out %out: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.strictconnect %out, %c0_ui1 : !firrtl.uint<1>
  }
  firrtl.module @constOutput(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    %c_out = firrtl.instance c @constOutChild(out out: !firrtl.uint<1>)
    %0 = firrtl.and %x, %c_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %z, %0 : !firrtl.uint<1>
    // CHECK: %[[C3_0:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[C3:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %z, %[[C3:.+]] : !firrtl.uint<1>
  }
}

// Optimizing this mux gives: z <= pad(UInt<2>(0), 4)
// Thus this checks that we then optimize that pad
//"ConstProp" should "optimize nested Expressions" in {
firrtl.circuit "optiMux"   {
  // CHECK-LABEL: firrtl.module @optiMux
  firrtl.module @optiMux(out %z: !firrtl.uint<4>) {
    %c1_ui = firrtl.constant 1 : !firrtl.uint
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %0 = firrtl.mux(%c1_ui, %c0_ui2, %c0_ui4) : (!firrtl.uint, !firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>
    // CHECK: %[[C4:.+]] = firrtl.constant 0 :
    // CHECK: firrtl.strictconnect %z, %[[C4]]
    firrtl.strictconnect %z, %0 : !firrtl.uint<4>
  }
}

firrtl.circuit "divFold"   {
  // CHECK-LABEL: firrtl.module @divFold
  firrtl.module @divFold(in %a: !firrtl.uint<8>, out %b: !firrtl.uint<8>) {
    %0 = firrtl.div %a, %a : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.strictconnect %b, %0 : !firrtl.uint<8>
    // CHECK: %[[C5:.+]] = firrtl.constant 1 : !firrtl.uint<8>
    // CHECK: firrtl.strictconnect %b, %[[C5]] : !firrtl.uint<8>
  }
}

//  "remove pads if the width is <= the width of the argument"
firrtl.circuit "removePad"   {
  // CHECK-LABEL: firrtl.module @removePad
  firrtl.module @removePad(in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
    %0 = firrtl.pad %x, 6 : (!firrtl.uint<8>) -> !firrtl.uint<8>
    // CHECK: firrtl.strictconnect %z, %x : !firrtl.uint<8>
    firrtl.strictconnect %z, %0 : !firrtl.uint<8>
  }
}

//"Registers async reset and a constant connection" should "NOT be removed
firrtl.circuit "asyncReset"   {
  // CHECK-LABEL: firrtl.module @asyncReset
  firrtl.module @asyncReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %en: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
    %c11_ui4 = firrtl.constant 11 : !firrtl.uint<4>
    %r = firrtl.regreset %clock, %reset, %c11_ui4  : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<4>, !firrtl.uint<8>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %0 = firrtl.mux(%en, %c0_ui4, %r) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.strictconnect %r, %0 : !firrtl.uint<8>
    firrtl.strictconnect %z, %r : !firrtl.uint<8>
    // CHECK: firrtl.strictconnect %r, %0 : !firrtl.uint<8>
    // CHECK: firrtl.strictconnect %z, %r : !firrtl.uint<8>
  }
}

//"propagation of signed expressions" should "have the correct signs"
firrtl.circuit "SignTester"   {
  // CHECK-LABEL: firrtl.module @SignTester
  firrtl.module @SignTester(out %ref: !firrtl.sint<3>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c0_si3 = firrtl.constant 0 : !firrtl.sint<3>
    %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
    %0 = firrtl.neg %c3_ui2 : (!firrtl.uint<2>) -> !firrtl.sint<3>
    %1 = firrtl.mux(%c0_ui1, %c0_si3, %0) : (!firrtl.uint<1>, !firrtl.sint<3>, !firrtl.sint<3>) -> !firrtl.sint<3>
    firrtl.strictconnect %ref, %1 : !firrtl.sint<3>
    // CHECK:  %[[C14:.+]] = firrtl.constant -3 : !firrtl.sint<3>
    // CHECK:  firrtl.strictconnect %ref, %[[C14]] : !firrtl.sint<3>
  }
}

//"addition of negative literals" should "be propagated"
firrtl.circuit "AddTester"   {
  // CHECK-LABEL: firrtl.module @AddTester
  firrtl.module @AddTester(out %ref: !firrtl.sint<2>) {
    %c-1_si1 = firrtl.constant -1 : !firrtl.sint<1>
    %0 = firrtl.add %c-1_si1, %c-1_si1 : (!firrtl.sint<1>, !firrtl.sint<1>) -> !firrtl.sint<2>
    firrtl.strictconnect %ref, %0 : !firrtl.sint<2>
    // CHECK:  %[[C15:.+]] = firrtl.constant -2 : !firrtl.sint<2>
    // CHECK:  firrtl.strictconnect %ref, %[[C15]]
  }
}

//"reduction of literals" should "be propagated"
firrtl.circuit "ConstPropReductionTester"   {
  // CHECK-LABEL: firrtl.module @ConstPropReductionTester
  firrtl.module @ConstPropReductionTester(out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>, out %out3: !firrtl.uint<1>) {
    %c-1_si2 = firrtl.constant -1 : !firrtl.sint<2>
    %0 = firrtl.xorr %c-1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
    firrtl.strictconnect %out1, %0 : !firrtl.uint<1>
    %1 = firrtl.andr %c-1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
    firrtl.strictconnect %out2, %1 : !firrtl.uint<1>
    %2 = firrtl.orr %c-1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
    firrtl.strictconnect %out3, %2 : !firrtl.uint<1>
    // CHECK:  %[[C16:.+]] = firrtl.constant 0
    // CHECK:  %[[C17:.+]] = firrtl.constant 1
    // CHECK:  firrtl.strictconnect %out1, %[[C16]]
    // CHECK:  firrtl.strictconnect %out2, %[[C17]]
    // CHECK:  firrtl.strictconnect %out3, %[[C17]]
  }
}

firrtl.circuit "TailTester"   {
  // CHECK-LABEL: firrtl.module @TailTester
  firrtl.module @TailTester(out %out: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c23_ui5 = firrtl.constant 23 : !firrtl.uint<5>
    %0 = firrtl.add %c0_ui1, %c23_ui5 : (!firrtl.uint<1>, !firrtl.uint<5>) -> !firrtl.uint<6>
    %_temp = firrtl.node droppable_name %0  : !firrtl.uint<6>
    %1 = firrtl.head %_temp, 3 : (!firrtl.uint<6>) -> !firrtl.uint<3>
    %_head_temp = firrtl.node droppable_name %1  : !firrtl.uint<3>
    %2 = firrtl.tail %_head_temp, 2 : (!firrtl.uint<3>) -> !firrtl.uint<1>
    firrtl.strictconnect %out, %2 : !firrtl.uint<1>
    // CHECK:  %[[C18:.+]] = firrtl.constant 0
    // CHECK:  firrtl.strictconnect %out, %[[C18]]
  }
}

//"tail of constants" should "be propagated"
firrtl.circuit "TailTester2"   {
  // CHECK-LABEL: firrtl.module @TailTester2
  firrtl.module @TailTester2(out %out: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c23_ui5 = firrtl.constant 23 : !firrtl.uint<5>
    %0 = firrtl.add %c0_ui1, %c23_ui5 : (!firrtl.uint<1>, !firrtl.uint<5>) -> !firrtl.uint<6>
    %_temp = firrtl.node droppable_name %0  : !firrtl.uint<6>
    %1 = firrtl.tail %_temp, 1 : (!firrtl.uint<6>) -> !firrtl.uint<5>
    %_tail_temp = firrtl.node droppable_name %1  : !firrtl.uint<5>
    %2 = firrtl.tail %_tail_temp, 4 : (!firrtl.uint<5>) -> !firrtl.uint<1>
    firrtl.strictconnect %out, %2 : !firrtl.uint<1>
    // CHECK:  %[[C21:.+]] = firrtl.constant 1
    // CHECK:  firrtl.strictconnect %out, %[[C21]]
  }
}

//"addition by zero width wires" should "have the correct widths"
firrtl.circuit "ZeroWidthAdd"   {
  // CHECK-LABEL: firrtl.module @ZeroWidthAdd
  firrtl.module @ZeroWidthAdd(in %x: !firrtl.uint<0>, out %y: !firrtl.uint<7>) {
    %c0_ui9 = firrtl.constant 0 : !firrtl.uint<9>
    %0 = firrtl.add %x, %c0_ui9 : (!firrtl.uint<0>, !firrtl.uint<9>) -> !firrtl.uint<10>
    %_temp = firrtl.node droppable_name %0  : !firrtl.uint<10>
    %1 = firrtl.cat %_temp, %_temp : (!firrtl.uint<10>, !firrtl.uint<10>) -> !firrtl.uint<20>
    %2 = firrtl.tail %1, 13 : (!firrtl.uint<20>) -> !firrtl.uint<7>
    firrtl.strictconnect %y, %2 : !firrtl.uint<7>
    // CHECK:  %[[C20:.+]] = firrtl.constant 0
    // CHECK:  firrtl.strictconnect %y, %[[C20]]
  }
}

//"Registers with constant reset and connection to the same constant" should "be replaced with that constant"
firrtl.circuit "regConstReset"   {
  // CHECK-LABEL: firrtl.module @regConstReset
  firrtl.module @regConstReset(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
    %c11_ui8 = firrtl.constant 11 : !firrtl.uint<8>
    %r = firrtl.regreset %clock, %reset, %c11_ui8  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    %0 = firrtl.mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.strictconnect %r, %0 : !firrtl.uint<8>
    firrtl.strictconnect %z, %r : !firrtl.uint<8>
    // CHECK: %[[C22:.+]] = firrtl.constant 11 
    // CHECK: firrtl.strictconnect %z, %[[C22]]
  }
}
