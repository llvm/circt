// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{body may only contain hw.typedecl operations}}
sv.package @invalid {
  sv.verbatim "wire x;"
}

// -----

// expected-error @+1 {{body may only contain hw.typedecl operations}}
sv.package.extern @invalid {
  sv.verbatim "wire x;"
}

// -----

// expected-error @+1 {{non-public packages require 'hw.verilogName'}}
sv.package.extern @private_types {} {sym_visibility = "private"}

// -----

// expected-error @+1 {{non-public packages require 'hw.verilogName'}}
sv.package.extern @nested_types {} {sym_visibility = "nested"}

// -----

// expected-error @+1 {{'hw.verilogName' must be a non-empty string}}
sv.package.extern @invalid_name {} {hw.verilogName = 1 : i32}

// -----

// expected-error @+1 {{'hw.verilogName' must be a non-empty string}}
sv.package.extern @empty_name {} {hw.verilogName = ""}

// -----

hw.module @invalid() {
  // expected-error @+1 {{expects parent op 'builtin.module'}}
  sv.package @nested {}
}

// -----

hw.module @invalid() {
  // expected-error @+1 {{expects parent op 'builtin.module'}}
  sv.package.extern @nested {}
}

// -----

// expected-error @+1 {{expects parent op to be a type scope, such as 'hw.type_scope' or 'sv.package'}}
hw.typedecl @unscoped : i8

// -----

sv.interface @not_a_type_scope {
  // expected-error @+1 {{expects parent op to be a type scope, such as 'hw.type_scope' or 'sv.package'}}
  hw.typedecl @unscoped : i8
}
