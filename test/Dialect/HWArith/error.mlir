// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @tooFewOps() {
  %c1_1 = hwarith.constant 1 : ui1
  // expected-error @+1 {{expected 2 operands but got 1}}
  %0 = hwarith.add %c1_1: (ui1) -> ui1
}

// -----

hw.module @tooManyOps() {
  %c1_1 = hwarith.constant 1 : ui1
  // expected-error @+1 {{expected 2 operands but got 3}}
  %0 = hwarith.add %c1_1, %c1_1, %c1_1: (ui1, ui1, ui1) -> ui2
}

// -----

hw.type_scope @ns {
  hw.typedecl @t : i4
}

hw.module @signlessAliasConstant() {
  // expected-error @+1 {{'hwarith.constant' op result #0 must be an arbitrary precision integer with signedness semantics or a type alias of one}}
  %0 = hwarith.constant 1 : i4 : !hw.typealias<@ns::@t, i4>
}

// -----

hw.type_scope @ns {
  hw.typedecl @t : si4
}

hw.module @aliasConstantTypeMismatch() {
  // expected-error @+1 {{'hwarith.constant' op value type 'ui4' doesn't match the canonical result type 'si4'}}
  %0 = hwarith.constant 1 : ui4 : !hw.typealias<@ns::@t, si4>
}
