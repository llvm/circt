// RUN: circt-opt %s --convert-verif-to-smt --split-input-file --verify-diagnostics

func.func @assert_with_unsupported_property_type(%arg0: !smt.bv<1>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to !ltl.sequence
  // expected-error @below {{failed to legalize operation 'verif.assert' that was explicitly marked illegal}}
  verif.assert %0 : !ltl.sequence
  return
}

// -----

func.func @assert_with_unsupported_property_type(%arg0: !smt.bv<1>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to !ltl.property
  // expected-error @below {{failed to legalize operation 'verif.assert' that was explicitly marked illegal}}
  verif.assert %0 : !ltl.property
  return
}
