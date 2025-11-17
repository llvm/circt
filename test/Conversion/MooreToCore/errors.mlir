// RUN: circt-opt %s --convert-moore-to-core --split-input-file --verify-diagnostics

func.func @invalidType() {
  // expected-error @below {{failed to legalize operation 'moore.variable'}}
  %var = moore.variable : <!moore.queue<string, 42>>

  return
}

// -----

func.func @unsupportedConversion() {
    %0 = moore.constant_string "Test" : i32
    // expected-error @below {{conversion result type is not currently supported}}
    // expected-error @below {{failed to legalize operation 'moore.conversion'}}
    %1 = moore.conversion %0 : !moore.i32 -> !moore.string
  return
}
