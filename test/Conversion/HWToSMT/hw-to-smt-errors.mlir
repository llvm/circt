// RUN: circt-opt --convert-hw-to-smt --split-input-file --verify-diagnostics %s

hw.module @zeroBitConstant() {
  // expected-error @below {{failed to legalize operation 'hw.constant' that was explicitly marked illegal}}
  %c0_i0 = hw.constant 0 : i0
  hw.output
}
