// RUN: cirt-opt %s -split-input-file -verify-diagnostics

func @test_constant() -> i32 {
  // expected-error @+1 {{firrtl.constant attribute bitwidth doesn't match return type}}
  %a = rtl.constant(42 : i12) : i32
  return %a : i32
}

// -----

func @test_extend(%arg0: i4) -> i4 {
  // expected-error @+1 {{extension must increase bitwidth of operand}}
  %a = rtl.sext %arg0 : i4, i4
  return %a : i4
}

// -----

func @test_trunc(%arg0: i4) -> i4 {
  // expected-error @+1 {{rtl.trunc must reduce bitwidth of operand}}
  %a = rtl.trunc %arg0 : i4, i4
  return %a : i4
}
