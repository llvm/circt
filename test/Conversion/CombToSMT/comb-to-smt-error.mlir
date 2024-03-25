// RUN: circt-opt %s --convert-comb-to-smt --split-input-file --verify-diagnostics

func.func @test(%a0: !smt.bv<32>, %a1: !smt.bv<32>) {
  %arg0 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  %arg1 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  // expected-error @below {{failed to legalize operation 'comb.icmp' that was explicitly marked illegal}}
  %14 = comb.icmp weq %arg0, %arg1 : i32
  return
}

// -----

func.func @test(%a0: !smt.bv<32>, %a1: !smt.bv<32>) {
  %arg0 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  %arg1 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  // expected-error @below {{failed to legalize operation 'comb.icmp' that was explicitly marked illegal}}
  %14 = comb.icmp ceq %arg0, %arg1 : i32
  return
}

// -----

func.func @test(%a0: !smt.bv<32>, %a1: !smt.bv<32>) {
  %arg0 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  %arg1 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  // expected-error @below {{failed to legalize operation 'comb.icmp' that was explicitly marked illegal}}
  %14 = comb.icmp wne %arg0, %arg1 : i32
  return
}

// -----

func.func @test(%a0: !smt.bv<32>, %a1: !smt.bv<32>) {
  %arg0 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  %arg1 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  // expected-error @below {{failed to legalize operation 'comb.icmp' that was explicitly marked illegal}}
  %14 = comb.icmp cne %arg0, %arg1 : i32
  return
}

// -----

func.func @zero_width_parity(%a0: !smt.bv<32>) {
  %arg0 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i0
  // Fails because "unable to convert type for operand #0, type was 'i0'"
  // expected-error @below {{failed to legalize operation 'comb.parity' that was explicitly marked illegal}}
  %0 = comb.parity %arg0 : i0
  return
}
