// RUN: circt-opt %s --convert-moore-to-core --split-input-file --verify-diagnostics

func.func @invalidType() {
  // expected-error @below {{failed to legalize operation 'moore.variable'}}
  %var = moore.variable : <!moore.queue<string, 42>>

  return
}

// -----

func.func @invalidStringContant() {
  // expected-error @below {{failed to legalize operation 'moore.string_constant'}}
  %str = moore.string_constant "Test" : i8

  return
}
