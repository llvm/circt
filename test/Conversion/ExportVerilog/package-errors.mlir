// RUN: circt-opt %s -split-input-file -export-verilog -verify-diagnostics -o /dev/null

sv.package @types {}
// expected-error @+1 {{unresolvable type reference}}
hw.module @UnknownPackage(in %word: !hw.typealias<@missing::@word, i8>) {}

// -----

sv.package @types {}
// expected-error @+1 {{unresolvable type reference}}
hw.module @UnknownDeclaration(in %word: !hw.typealias<@types::@missing, i8>) {}

// -----

sv.package @types {
  hw.typedecl @word : i8
}
// expected-error @+1 {{declared type did not match aliased type}}
hw.module @WrongInnerType(in %word: !hw.typealias<@types::@word, i16>) {}

// -----

sv.package @types {
  hw.typedecl @word : i8
}
// A package cannot be emitted into more than one file.
// expected-error @+1 {{packages can be emitted to a single file}}
sv.package @duplicated {}
emit.file "first.sv" { emit.ref @duplicated }
emit.file "second.sv" { emit.ref @duplicated }
