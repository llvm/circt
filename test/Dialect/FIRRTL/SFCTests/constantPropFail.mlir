// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop), canonicalize{top-down region-simplify})'  %s | FileCheck %s

// These depend on more than constant prop.  They need to move.
// XFAIL: *

// Registers with no reset or connections" should "be replaced with constant zero
firrtl.circuit "uninitSelfReg"   {
  // CHECK-LABEL: firrtl.module @uninitSelfReg
  firrtl.module @uninitSelfReg(in %clock: !firrtl.clock, out %z: !firrtl.uint<8>) {
    %r = firrtl.reg %clock  :  !firrtl.uint<8>
    firrtl.connect %r, %r : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
    // CHECK: firrtl.strictconnect %z, %invalid_ui8 : !firrtl.uint<8>
  }
}

// "pad zero when constant propping a register replaced with zero"
firrtl.circuit "padZeroReg"   {
  // CHECK-LABEL: firrtl.module @padZeroReg
  firrtl.module @padZeroReg(in %clock: !firrtl.clock, out %z: !firrtl.uint<16>) {
      %_r = firrtl.reg droppable_name %clock  :  !firrtl.uint<8>
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      %0 = firrtl.or %_r, %c0_ui1 : (!firrtl.uint<8>, !firrtl.uint<1>) -> !firrtl.uint<8>
      firrtl.connect %_r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
      %c171_ui8 = firrtl.constant 171 : !firrtl.uint<8>
      %_n = firrtl.node droppable_name %c171_ui8  : !firrtl.uint<8>
      %1 = firrtl.cat %_n, %_r : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<16>
      firrtl.connect %z, %1 : !firrtl.uint<16>, !firrtl.uint<16>
    // CHECK: %[[TMP:.+]] = firrtl.constant 43776 : !firrtl.uint<16>
    // CHECK-NEXT: firrtl.strictconnect %z, %[[TMP]] : !firrtl.uint<16>
  }
}

