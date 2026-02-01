// RUN: circt-opt %s --convert-moore-to-core --split-input-file --verify-diagnostics

func.func @invalidType() {
  // expected-error @below {{failed to legalize operation 'moore.variable'}}
  %var = moore.variable : <!moore.queue<string, 42>>

  return
}

// -----

// expected-error @below {{port '"queue_port"' has unsupported type '!moore.queue<i32, 10>' that cannot be converted to hardware type}}
// expected-error @below {{failed to legalize}}
moore.module @UnsupportedInputPortType(in %queue_port : !moore.queue<i32, 10>) {
  moore.output
}

// -----

// expected-error @below {{port '"data"' has unsupported type '!moore.queue<i32, 10>' that cannot be converted to hardware type}}
// expected-error @below {{failed to legalize}}
moore.module @MixedPortsWithUnsupported(in %valid : !moore.l1, in %data : !moore.queue<i32, 10>, out out : !moore.l1) {
  moore.output %valid : !moore.l1
}

// -----
