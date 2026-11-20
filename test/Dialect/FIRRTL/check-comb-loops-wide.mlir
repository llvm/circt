// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-comb-loops))' %s | FileCheck %s

// Every one of the 32768 output fields depends on every input field. A check
// that walks each output field's cone separately, or records each
// (output, input) field pair, does about 10^9 steps here and runs out of
// memory or time; sharing one set of input fields keeps it linear.

// CHECK: firrtl.circuit "Wide"
firrtl.circuit "Wide" {
  firrtl.module @Wide(in %a: !firrtl.vector<uint<1>, 32768>, out %b: !firrtl.vector<uint<1>, 32768>) {
    %u = firrtl.bitcast %a : (!firrtl.vector<uint<1>, 32768>) -> !firrtl.uint<32768>
    %n = firrtl.not %u : (!firrtl.uint<32768>) -> !firrtl.uint<32768>
    %v = firrtl.bitcast %n : (!firrtl.uint<32768>) -> !firrtl.vector<uint<1>, 32768>
    firrtl.connect %b, %v : !firrtl.vector<uint<1>, 32768>, !firrtl.vector<uint<1>, 32768>
  }
}
