// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-specialize-layers))' -split-input-file -verify-diagnostics %s

// expected-error @below {{unknown layer @A}}
firrtl.circuit "Test" attributes {
  enable_layers = [@A]
} {
  firrtl.module @Test() {}
}

// -----

// expected-error @below {{unknown layer @A}}
firrtl.circuit "Test" attributes {
  disable_layers = [@A]
} {
  firrtl.module @Test() {}
}

// -----

// expected-error @below {{layer @A both enabled and disabled}}
firrtl.circuit "Test" attributes {
  enable_layers = [@A],
  disable_layers = [@A]
} {
  firrtl.layer @A bind { }
  firrtl.module @Test() {}
}
