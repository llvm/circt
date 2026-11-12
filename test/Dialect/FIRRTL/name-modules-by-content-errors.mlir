// RUN: circt-opt -verify-diagnostics -split-input-file -pass-pipeline='builtin.module(firrtl.circuit(firrtl-name-modules-by-content))' %s

// Two identical modules that were not deduplicated would get the same name.
firrtl.circuit "Identical" {
  // expected-note @below {{the other module}}
  firrtl.module private @Same_1(in %in: !firrtl.uint<1>) {}
  // expected-error @below {{they are identical, and were not deduplicated}}
  firrtl.module private @Same_2(in %in: !firrtl.uint<1>) {}
  firrtl.module @Identical() {
    %a_in = firrtl.instance a @Same_1(in in: !firrtl.uint<1>)
    %b_in = firrtl.instance b @Same_2(in in: !firrtl.uint<1>)
  }
}

// -----

// A new name may not take the name of another module.
firrtl.circuit "Taken" {
  // expected-error @below {{cannot rename module 'Taken_1' to 'Taken_a32eff6e': the name is taken}}
  firrtl.module private @Taken_1() {}
  // expected-note @below {{by this}}
  firrtl.module private @Taken_a32eff6e() {}
  firrtl.module @Taken() {
    firrtl.instance a @Taken_1()
    firrtl.instance b @Taken_a32eff6e()
  }
}
