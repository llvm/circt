// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop))' -verify-diagnostics --split-input-file %s


// https://github.com/llvm/circt/issues/1187
// This shouldn't crash.
firrtl.circuit "Issue1187"  {
  firrtl.module @Issue1187(in %divisor: !firrtl.uint<1>, out %result: !firrtl.uint<0>) {
    %dividend = firrtl.wire  : !firrtl.uint<0>
    %invalid_ui0 = firrtl.invalidvalue : !firrtl.uint<0>
    firrtl.connect %dividend, %invalid_ui0 : !firrtl.uint<0>, !firrtl.uint<0>
    %0 = firrtl.div %dividend, %divisor : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
    firrtl.connect %result, %0 : !firrtl.uint<0>, !firrtl.uint<0>
  }
}
