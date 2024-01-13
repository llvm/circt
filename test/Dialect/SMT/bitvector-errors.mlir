// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{bit-vector must have at least a width of one}}
func.func @at_least_size_one(%arg0: !smt.bv<0>) {
  return
}

// -----

func.func @attr_type_and_return_type_match() {
  // expected-error @below {{smt.bv.constant attribute bitwidth doesn't match return type}}
  %c0_bv32 = "smt.bv.constant"() <{value = #smt.bv<"#b0"> : !smt.bv<1>}> : () -> !smt.bv<32>
  return
}

// -----

func.func @implicit_constant_type_and_explicit_type_match() {
  // expected-error @below {{expected type for constant does not match explicitly provided attribute type, got '!smt.bv<2>', expected '!smt.bv<1>'}}
  %c0_bv2 = "smt.bv.constant"() <{value = #smt.bv<"#b0"> : !smt.bv<2>}> : () -> !smt.bv<1>
  return
}

// -----

func.func @invalid_bitvector_attr() {
  // expected-error @below {{expected at least one digit}}
  smt.bv.constant #smt.bv<"#b">
}

// -----

func.func @invalid_bitvector_attr() {
  // expected-error @below {{expected either 'b' or 'x'}}
  smt.bv.constant #smt.bv<"#c0">
}

// -----

func.func @invalid_bitvector_attr() {
  // expected-error @below {{expected '#'}}
  smt.bv.constant #smt.bv<"b">
}
