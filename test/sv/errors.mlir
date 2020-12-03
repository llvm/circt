// RUN: circt-opt %s -split-input-file -verify-diagnostics

rtl.module @test_extract_error() {
  %a = rtl.constant(42 : i12) : i32
  // expected-error @+1 {{custom op 'sv.extract' Expected extractable type, got 'i32'}}
  %b = sv.extract @nofield %a : i32
}

// -----
