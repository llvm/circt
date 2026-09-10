// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s -mlir-print-op-generic | circt-opt | FileCheck %s

// CHECK-LABEL: sv.package @empty {
// CHECK-NEXT: }
sv.package @empty {}

// CHECK-LABEL: sv.package @types {
sv.package @types {
  // CHECK: hw.typedecl @word, "word_t" : i8
  hw.typedecl @word, "word_t" : i8
  // CHECK: hw.typedecl @record : !hw.struct<data: !hw.typealias<@types::@word, i8>, valid: i1>
  hw.typedecl @record : !hw.struct<data: !hw.typealias<@types::@word, i8>, valid: i1>
  // CHECK: hw.typedecl @words : !hw.array<4xtypealias<@types::@word, i8>>
  hw.typedecl @words : !hw.array<4x!hw.typealias<@types::@word, i8>>
  // CHECK: hw.typedecl @unpacked_words : !hw.uarray<2xtypealias<@types::@word, i8>>
  hw.typedecl @unpacked_words : !hw.uarray<2x!hw.typealias<@types::@word, i8>>
  // CHECK: hw.typedecl @choice : !hw.union<a: i8, b: i16>
  hw.typedecl @choice : !hw.union<a: i8, b: i16>
  // CHECK: hw.typedecl @state : !hw.enum<idle, busy>
  hw.typedecl @state : !hw.enum<idle, busy>
  // CHECK: hw.typedecl @hidden : i1 {sym_visibility = "private"}
  hw.typedecl @hidden : i1 {sym_visibility = "private"}
}

// CHECK-LABEL: sv.package @private_types {
// CHECK-NEXT: } {sym_visibility = "private"}
sv.package @private_types {} {sym_visibility = "private"}

// Package and legacy scopes share the same alias type system.
// CHECK-LABEL: hw.type_scope @legacy {
hw.type_scope @legacy {
  // CHECK: hw.typedecl @word : !hw.typealias<@types::@word, i8>
  hw.typedecl @word : !hw.typealias<@types::@word, i8>
}

// CHECK-LABEL: sv.package @aliases {
sv.package @aliases {
  // CHECK: hw.typedecl @word : !hw.typealias<@legacy::@word, !hw.typealias<@types::@word, i8>>
  hw.typedecl @word : !hw.typealias<@legacy::@word, !hw.typealias<@types::@word, i8>>
}

// CHECK-LABEL: hw.module @use_package(
hw.module @use_package(
  in %arg: !hw.typealias<@types::@record, !hw.struct<data: !hw.typealias<@types::@word, i8>, valid: i1>>,
  out word: !hw.typealias<@types::@word, i8>) {
  // CHECK: [[WORD:%.+]] = hw.struct_extract %arg["data"]
  %word = hw.struct_extract %arg["data"] : !hw.typealias<@types::@record, !hw.struct<data: !hw.typealias<@types::@word, i8>, valid: i1>>
  // CHECK: hw.output [[WORD]] : !hw.typealias<@types::@word, i8>
  hw.output %word : !hw.typealias<@types::@word, i8>
}
