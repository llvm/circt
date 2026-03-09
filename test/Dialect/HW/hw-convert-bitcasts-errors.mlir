// RUN: circt-opt -verify-diagnostics --hw-convert-bitcasts=allow-partial-conversion=false --split-input-file %s

hw.module @unsupported(in %i: i8, out o : i8) {
  // expected-error @+1 {{has unsupported output type}}
  %a = hw.bitcast %i : (i8) -> !hw.union<foo: i8, bar: i8>
  // expected-error @+1 {{has unsupported input type}}
  %o = hw.bitcast %a : (!hw.union<foo: i8, bar: i8>) -> i8
  hw.output %o : i8
}
