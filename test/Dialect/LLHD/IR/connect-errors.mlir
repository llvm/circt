// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-note @+1 {{prior use here}}
hw.module @connect_different_types(inout %in: i8, inout %out: i32) {
  // expected-error @+1 {{use of value '%out' expects different type}}
  llhd.con %in, %out : !hw.inout<i8>
}

// -----

hw.module @connect_non_signals(inout %in: i32, inout %out: i32) {
  %0 = llhd.prb %in : !hw.inout<i32>
  %1 = llhd.prb %out : !hw.inout<i32>
  // expected-error @+1 {{'llhd.con' op operand #0 must be InOutType, but got 'i32'}}
  llhd.con %0, %1 : i32
}
