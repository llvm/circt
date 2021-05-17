// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-infer-widths)' --verify-diagnostics --split-input-file %s

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %clock : !firrtl.clock) {
    // expected-error @+1 {{'firrtl.reg' op not supported in width inference}}
    %0 = firrtl.reg %clock : (!firrtl.clock) -> !firrtl.uint<4>
    %1 = firrtl.wire : !firrtl.uint
    firrtl.connect %1, %0 : !firrtl.uint, !firrtl.uint<4>
  }
}
