// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop))' -verify-diagnostics --split-input-file %s


// https://github.com/llvm/circt/issues/1187
// This shouldn't crash.
firrtl.circuit "Issue1187"  {
  firrtl.module @Issue1187(in %divisor: !firrtl.uint<1>, out %result: !firrtl.uint<0>) {
    %dividend = firrtl.wire  : !firrtl.uint<0>
    %invalid_ui0 = firrtl.invalidvalue : !firrtl.uint<0>
    firrtl.matchingconnect %dividend, %invalid_ui0 : !firrtl.uint<0>
    %0 = firrtl.div %dividend, %divisor : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
    firrtl.matchingconnect %result, %0 : !firrtl.uint<0>
  }
}

// -----

// https://github.com/llvm/circt/issues/4456
// This shouldn't crash.
// The fold is what should be tested but need IMCP to drive it.
firrtl.circuit "Issue4456"  {
  firrtl.module @Issue4456(in %i: !firrtl.sint<0>, out %o: !firrtl.uint<4>) {
    %c0_si4 = firrtl.constant 0 : !firrtl.sint<4>
    %0 = firrtl.cat %i, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<4>
    firrtl.matchingconnect %o, %0 : !firrtl.uint<4>
  }
}

// -----

// https://github.com/llvm/circt/issues/11635
// This shouldn't crash.
firrtl.circuit "Issue11635" {
  firrtl.module private @Child(in %i: !firrtl.enum<A, B: uint<8>>, out %o: !firrtl.enum<A, B: uint<8>>) {
    firrtl.matchingconnect %o, %i : !firrtl.enum<A, B: uint<8>>
  }

  firrtl.module @Issue11635(in %i: !firrtl.enum<A, B: uint<8>>, out %o: !firrtl.enum<A, B: uint<8>>) {
    %c_i, %c_o = firrtl.instance c @Child(in i: !firrtl.enum<A, B: uint<8>>, out o: !firrtl.enum<A, B: uint<8>>)
    firrtl.matchingconnect %c_i, %i : !firrtl.enum<A, B: uint<8>>
    firrtl.matchingconnect %o, %c_o : !firrtl.enum<A, B: uint<8>>
  }
}

// -----

// https://github.com/llvm/circt/issues/11635
// This shouldn't crash.
firrtl.circuit "Issue11635" {
  firrtl.module @Issue11635(out %o: !firrtl.enum<A, B: uint<8>>) {
    %c3_ui8 = firrtl.constant 3 : !firrtl.uint<8>
    %e = firrtl.enumcreate B(%c3_ui8) : (!firrtl.uint<8>) -> !firrtl.enum<A, B: uint<8>>
    %w = firrtl.wire : !firrtl.enum<A, B: uint<8>>
    firrtl.matchingconnect %w, %e : !firrtl.enum<A, B: uint<8>>
    firrtl.matchingconnect %o, %w : !firrtl.enum<A, B: uint<8>>
  }
}
