// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{bit-vector must have at least a width of one}}
func.func @at_least_size_one(%arg0: !smt.bv<0>) {
  return
}

// -----

func.func @attr_type_and_return_type_match() {
  // expected-error @below {{inferred type(s) '!smt.bv<1>' are incompatible with return type(s) of operation '!smt.bv<32>'}}
  // expected-error @below {{failed to infer returned types}}
  %c0_bv32 = "smt.bv.constant"() <{value = #smt.bv<0> : !smt.bv<1>}> : () -> !smt.bv<32>
  return
}

// -----

func.func @invalid_bitvector_attr() {
  // expected-error @below {{explicit bit-vector type required}}
  smt.bv.constant #smt.bv<5>
}

// -----

func.func @invalid_bitvector_attr() {
  // expected-error @below {{integer value out of range for given bit-vector type}}
  smt.bv.constant #smt.bv<32> : !smt.bv<2>
}

// -----

func.func @invalid_bitvector_attr() {
  // expected-error @below {{integer value out of range for given bit-vector type}}
  smt.bv.constant #smt.bv<-4> : !smt.bv<2>
}
