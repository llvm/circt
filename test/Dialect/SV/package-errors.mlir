// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{body may only contain hw.typedecl operations}}
sv.package @invalid {
  sv.verbatim "wire x;"
}

// -----

// expected-error @+1 {{body may only contain hw.typedecl operations}}
sv.package @invalid {
  hw.type_scope @nested {}
}

// -----

// expected-error @+1 {{body may only contain hw.typedecl operations}}
sv.package @invalid {
  hw.module.extern @module()
}

// -----

hw.module @invalid() {
  // expected-error @+1 {{expects parent op 'builtin.module'}}
  sv.package @nested {}
}

// -----

// expected-error @+1 {{region should have no arguments}}
"sv.package"() ({
^bb0(%arg: i1):
}) {sym_name = "invalid"} : () -> ()

// -----

// expected-error @+1 {{region #0 ('body') failed to verify constraint: region with 1 blocks}}
"sv.package"() ({
}) {sym_name = "invalid"} : () -> ()

// -----

sv.package @duplicate {
  // expected-note @+1 {{see existing symbol definition here}}
  hw.typedecl @word : i8
  // expected-error @+1 {{redefinition of symbol named 'word'}}
  hw.typedecl @word : i16
}

// -----

// expected-note @+1 {{see existing symbol definition here}}
sv.package @duplicate {}
// expected-error @+1 {{redefinition of symbol named 'duplicate'}}
sv.package @duplicate {}

// -----

// expected-error @+1 {{expects parent op to be a type scope, such as 'hw.type_scope' or 'sv.package'}}
hw.typedecl @unscoped : i8

// -----

sv.interface @not_a_type_scope {
  // expected-error @+1 {{expects parent op to be a type scope, such as 'hw.type_scope' or 'sv.package'}}
  hw.typedecl @unscoped : i8
}
