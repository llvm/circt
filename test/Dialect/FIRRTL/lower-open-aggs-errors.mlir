// RUN: circt-opt --pass-pipeline="builtin.module(firrtl.circuit(firrtl-lower-open-aggs))" %s --split-input-file --verify-diagnostics

firrtl.circuit "Symbol" {
  // expected-error @below {{symbol found on aggregate with no HW}}
  firrtl.module @Symbol(out %r : !firrtl.openbundle<p: probe<uint<1>>> sym @bad) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %ref = firrtl.ref.send %zero : !firrtl.uint<1>
    %r_p = firrtl.opensubfield %r[p] : !firrtl.openbundle<p: probe<uint<1>>>
    firrtl.ref.define %r_p, %ref : !firrtl.probe<uint<1>>
  }
}
