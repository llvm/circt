// RUN: circt-opt %s --split-input-file --verify-diagnostics

func.func @int2bv_neg_width() {
  %0 = smt.int.constant 5
  // expected-error @below {{op attribute 'width' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
  %int2bv = smt.int2bv %0 width -1 : !smt.bv<-1>
  return
}

// -----

func.func @int2bv_width_mismatch() {
  %0 = smt.int.constant 5
  // expected-error @below {{given width and return type width must match}}
  %int2bv = smt.int2bv %0 width 4 : !smt.bv<8>
  return
}
