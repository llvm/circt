// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{body may only contain hw.typedecl operations}}
sv.package @invalid {
  sv.verbatim "wire x;"
}

// -----

hw.module @invalid() {
  // expected-error @+1 {{expects parent op 'builtin.module'}}
  sv.package @nested {}
}

// -----

// expected-error @+1 {{expects parent op to be a type scope, such as 'hw.type_scope' or 'sv.package'}}
hw.typedecl @unscoped : i8

// -----

sv.interface @not_a_type_scope {
  // expected-error @+1 {{expects parent op to be a type scope, such as 'hw.type_scope' or 'sv.package'}}
  hw.typedecl @unscoped : i8
}
