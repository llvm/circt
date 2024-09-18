// RUN: circt-opt -verify-diagnostics -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-infer-rw)))' %s

firrtl.circuit "InferReadWriteErrors" {

  firrtl.module public @InferReadWriteErrors() {}

  firrtl.module @ContainsWhen() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // expected-error @below {{is unsupported by InferReadWrite}}
    firrtl.when %c0_ui1 : !firrtl.uint<1> {}
  }

}
