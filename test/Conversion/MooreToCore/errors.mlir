// RUN: circt-opt %s --convert-moore-to-core --split-input-file --verify-diagnostics

func.func @invalidType() {
  // expected-error @below {{failed to legalize operation 'moore.variable'}}
  %var = moore.variable : <!moore.queue<string, 42>>

  return
}

// -----

func.func @invalidStringContant() {
  // expected-error @below {{hw.constant attribute bitwidth doesn't match return type}}
  %str = moore.string_constant "Test" : i8

  return
}
