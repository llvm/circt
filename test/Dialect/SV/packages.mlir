// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: sv.package @empty {
// CHECK-NEXT: }
sv.package @empty {}

// CHECK-LABEL: sv.package @types {
sv.package @types {
  // CHECK: hw.typedecl @word, "word_t" : i8
  hw.typedecl @word, "word_t" : i8
}

// CHECK-LABEL: sv.package @private_types {
// CHECK-NEXT: } {sym_visibility = "private"}
sv.package @private_types {} {sym_visibility = "private"}

// CHECK-LABEL: sv.package.extern @external_empty {
// CHECK-NEXT: }
sv.package.extern @external_empty {}

// CHECK-LABEL: sv.package.extern @external_types {
// CHECK-NEXT: hw.typedecl @word, "word_t" : i8
// CHECK-NEXT: hw.typedecl @local_word : !hw.typealias<@types::@word, i8>
// CHECK-NEXT: } {hw.verilogName = "ExternalTypes", sym_visibility = "private"}
sv.package.extern @external_types {
  hw.typedecl @word, "word_t" : i8
  hw.typedecl @local_word : !hw.typealias<@types::@word, i8>
} {hw.verilogName = "ExternalTypes", sym_visibility = "private"}

// CHECK-LABEL: hw.module @use_package(
hw.module @use_package(
  in %arg: !hw.typealias<@types::@word, i8>,
  out word: !hw.typealias<@types::@word, i8>) {
  // CHECK: hw.output %arg : !hw.typealias<@types::@word, i8>
  hw.output %arg : !hw.typealias<@types::@word, i8>
}

// CHECK-LABEL: hw.module @use_external_package(
hw.module @use_external_package(
  in %arg: !hw.typealias<@external_types::@word, i8>,
  out word: !hw.typealias<@external_types::@word, i8>) {
  // CHECK: hw.output %arg : !hw.typealias<@external_types::@word, i8>
  hw.output %arg : !hw.typealias<@external_types::@word, i8>
}
