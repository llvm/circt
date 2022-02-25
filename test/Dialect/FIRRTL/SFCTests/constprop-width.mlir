// RUN: circt-opt -canonicalize='top-down=true region-simplify=true' -pass-pipeline='firrtl.circuit(firrtl-infer-widths, firrtl-imconstprop)' -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s
// github.com/chipsalliance/firrtl: test/scala/firrtlTests/ConstantPropagationTests.scala

//"pad constant connections to wires when propagating"
firrtl.circuit "padConstWire"   {
  // CHECK-LABEL: firrtl.module @padConstWire
  firrtl.module @padConstWire(out %z: !firrtl.uint<16>) {
    %w_a = firrtl.wire  : !firrtl.uint<8>
    %w_b = firrtl.wire  : !firrtl.uint<8>
    %c3_ui2 = firrtl.constant 3 : !firrtl.uint<3>
    firrtl.connect %w_a, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<3>
    firrtl.connect %w_b, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<3>
    %0 = firrtl.cat %w_a, %w_b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<16>
    firrtl.strictconnect %z, %0 : !firrtl.uint<16>
    // CHECK: %[[C6:.+]] = firrtl.constant 771 : !firrtl.uint<16>
    // CHECK-NEXT: firrtl.strictconnect %z, %[[C6]] : !firrtl.uint<16>
  }
}

// "pad constant connections to registers when propagating"
firrtl.circuit "padConstReg"   {
  // CHECK-LABEL: firrtl.module @padConstReg
  firrtl.module @padConstReg(in %clock: !firrtl.clock, out %z: !firrtl.uint<16>) {
    %r_a = firrtl.reg %clock  :  !firrtl.uint<8>
    %r_b = firrtl.reg %clock  :  !firrtl.uint<8>
    %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
    firrtl.connect %r_a, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<2>
    firrtl.connect %r_b, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<2>
    %0 = firrtl.cat %r_a, %r_b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<16>
    firrtl.strictconnect %z, %0 : !firrtl.uint<16>
    // CHECK: %[[C6:.+]] = firrtl.constant 771 : !firrtl.uint<16>
    // CHECK-NEXT: firrtl.strictconnect %z, %[[C6]] : !firrtl.uint<16>
  }
}

// should "pad constant connections to outputs when propagating"
firrtl.circuit "padConstOut"   {
  firrtl.module @padConstOutChild(out %x: !firrtl.uint<8>) {
    %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
    firrtl.connect %x, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<2>
  }
  // CHECK-LABEL: firrtl.module @padConstOut
  firrtl.module @padConstOut(out %z: !firrtl.uint<16>) {
    %c_x = firrtl.instance c @padConstOutChild(out x: !firrtl.uint<8>)
    %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
    %0 = firrtl.cat %c3_ui2, %c_x : (!firrtl.uint<2>, !firrtl.uint<8>) -> !firrtl.uint<10>
    // CHECK: %[[C8:.+]] = firrtl.constant 771 : !firrtl.uint<16>
    // CHECK: firrtl.strictconnect %z, %[[C8]] : !firrtl.uint<16>
    firrtl.connect %z, %0 : !firrtl.uint<16>, !firrtl.uint<10>
  }
}

// "pad constant connections to submodule inputs when propagating"
firrtl.circuit "padConstIn"   {
  // CHECK-LABEL: firrtl.module @padConstInChild
  firrtl.module @padConstInChild(in %x: !firrtl.uint<8>, out %y: !firrtl.uint<16>) {
    %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
    %0 = firrtl.cat %c3_ui2, %x : (!firrtl.uint<2>, !firrtl.uint<8>) -> !firrtl.uint<10>
    // CHECK: %[[C9:.+]] = firrtl.constant 771 : !firrtl.uint<16>
    // CHECK: firrtl.strictconnect %y, %[[C9]] : !firrtl.uint<16>
    firrtl.connect %y, %0 : !firrtl.uint<16>, !firrtl.uint<10>
  }
  // CHECK-LABEL: firrtl.module @padConstIn
  firrtl.module @padConstIn(out %z: !firrtl.uint<16>) {
    %c_x, %c_y = firrtl.instance c @padConstInChild(in x: !firrtl.uint<8>, out y: !firrtl.uint<16>)
    %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
    firrtl.connect %c_x, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<2>
    firrtl.strictconnect %z, %c_y : !firrtl.uint<16>
    // CHECK: %[[C10:.+]] = firrtl.constant 771 : !firrtl.uint<16>
    // CHECK: firrtl.strictconnect %z, %[[C10]] : !firrtl.uint<16>
  }
}
