// RUN: circt-opt %s --convert-moore-to-core --split-input-file --verify-diagnostics

func.func @invalidType() {
  // expected-error @below {{failed to legalize operation 'moore.variable'}}
  %var = moore.variable : <!moore.assoc_array<i32, i32>>

  return
}

// -----

// expected-error @below {{port '"queue_port"' has unsupported type '!moore.assoc_array<i32, string>' that cannot be converted to hardware type}}
// expected-error @below {{failed to legalize}}
moore.module @UnsupportedInputPortType(in %queue_port : !moore.assoc_array<i32, string>) {
  moore.output
}

// -----

// expected-error @below {{port '"data"' has unsupported type '!moore.assoc_array<i32, string>' that cannot be converted to hardware type}}
// expected-error @below {{failed to legalize}}
moore.module @MixedPortsWithUnsupported(in %valid : !moore.l1, in %data : !moore.assoc_array<i32, string>, out out : !moore.l1) {
  moore.output %valid : !moore.l1
}

// -----
