// RUN: circt-opt %s -split-input-file -verify-diagnostics

// Aliases with the same underlying type remain distinct across packages.
sv.package @first {
  hw.typedecl @word : i8
}
sv.package @second {
  hw.typedecl @word : i8
}

hw.module @different_packages(
    in %arg: !hw.typealias<@first::@word, i8>,
    out result: !hw.typealias<@second::@word, i8>) {
  // expected-error @+1 {{output types must match module}}
  hw.output %arg : !hw.typealias<@first::@word, i8>
}

// -----

// Aliases are not interchangeable with their underlying types.
sv.package @types {
  hw.typedecl @word : i8
}

hw.module @underlying_type(
    in %arg: !hw.typealias<@types::@word, i8>, out result: i8) {
  // expected-error @+1 {{output types must match module}}
  hw.output %arg : !hw.typealias<@types::@word, i8>
}

// -----

// A legacy scope and a package do not merge type identities.
hw.type_scope @legacy {
  hw.typedecl @word : i8
}
sv.package @types {
  hw.typedecl @word : i8
}

hw.module @different_scopes(
    in %arg: !hw.typealias<@legacy::@word, i8>,
    out result: !hw.typealias<@types::@word, i8>) {
  // expected-error @+1 {{output types must match module}}
  hw.output %arg : !hw.typealias<@legacy::@word, i8>
}
