// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-note @+1 {{prior use here}}
llhd.entity @connect_different_types(%in: !hw.inout<i8>) -> (%out: !hw.inout<i32>) {
  // expected-error @+1 {{use of value '%out' expects different type}}
  llhd.con %in, %out : !hw.inout<i8>
}

// -----

llhd.entity @connect_non_signals(%in: !hw.inout<i32>) -> (%out: !hw.inout<i32>) {
  %0 = llhd.prb %in : !hw.inout<i32>
  %1 = llhd.prb %out : !hw.inout<i32>
  // expected-error @+1 {{'llhd.con' op operand #0 must be InOutType}}
  llhd.con %0, %1 : i32
}
