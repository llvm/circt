// RUN: circt-translate --export-aiger --split-input-file %s --emit-text-format | FileCheck %s --check-prefix=WITH_SYMBOLS
// RUN: circt-translate --export-aiger --split-input-file --exclude-symbol-table %s --emit-text-format | FileCheck %s --check-prefix=NO_SYMBOLS

// Test symbol table export
// WITH_SYMBOLS-LABEL: aag 6 4 1 1 1
// WITH_SYMBOLS:       i0 input_a
// WITH_SYMBOLS-NEXT:  i1 input_b
// WITH_SYMBOLS-NEXT:  i2 input_c[0]
// WITH_SYMBOLS-NEXT:  i3 input_c[1]
// WITH_SYMBOLS-NEXT:  l0 my_register
// WITH_SYMBOLS-NEXT:  o0 output_result

// NO_SYMBOLS-LABEL: aag 6 4 1 1 1
// NO_SYMBOLS-NOT: i0
// NO_SYMBOLS-NOT: l0
// NO_SYMBOLS-NOT: o0
hw.module @symbol_test(in %input_a: i1, in %input_b: i1, in %input_c: i2, in %clk: !seq.clock, out output_result: i1) {
  %and_result = synth.aig.and_inv %input_b, %input_a : i1
  %my_register = seq.compreg %and_result, %clk : i1
  hw.output %my_register : i1
}
