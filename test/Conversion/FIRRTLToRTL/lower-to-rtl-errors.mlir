// RUN: circt-opt -lower-firrtl-to-rtl %s -verify-diagnostics

module attributes {firrtl.mainModule = "Simple"} {

  // https://github.com/llvm/circt/issues/593
  rtl.module @InvalidBundle() {
    // expected-error @+1 {{unsupported type}}
    %0 = firrtl.invalidvalue : !firrtl.bundle<inp_d: uint<14>>
    rtl.output
  }

}