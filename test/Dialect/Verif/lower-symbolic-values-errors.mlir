// RUN: circt-opt --verif-lower-symbolic-values=mode=extmodule --split-input-file --verify-diagnostics %s

hw.module @Foo() {
  // expected-error @below {{symbolic value bit width unknown}}
  verif.symbolic_value : !hw.string
}
