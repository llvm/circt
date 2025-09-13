// RUN: circt-translate --export-aiger %s --split-input-file --emit-text-format | FileCheck %s


// Test basic module with simple inputs and outputs
// CHECK-LABEL: aag 2 2 0 2 0
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 2
// CHECK-NEXT: 4
hw.module @basic_io(in %a: i1, in %b: i1, out x: i1, out y: i1) {
  hw.output %a, %b : i1, i1
}

// -----

// Test AND gate
// CHECK-LABEL: aag 3 2 0 1 1
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 6 2 4
hw.module @and_gate(in %a: i1, in %b: i1, out result: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

// -----

// Test AND gate with inversion
// CHECK-LABEL: aag 3 2 0 1 1
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 6 3 5
hw.module @and_gate_inverted(in %a: i1, in %b: i1, out result: i1) {
  %0 = synth.aig.and_inv not %a, not %b : i1
  hw.output %0 : i1
}

// -----

// Test single input inverter
// CHECK-LABEL: aag 1 1 0 1 0
// CHECK-NEXT: 2
// CHECK-NEXT: 3
hw.module @inverter(in %a: i1, out result: i1) {
  %0 = synth.aig.and_inv not %a : i1
  hw.output %0 : i1
}

// -----

// Test constants
// CHECK-LABEL: aag 0 0 0 2 0
// CHECK-NEXT: 0
// CHECK-NEXT: 1
hw.module @constants(out false_out: i1, out true_out: i1) {
  %false = hw.constant false
  %true = hw.constant true
  hw.output %false, %true : i1, i1
}

// -----

// Test register (latch)
// CHECK-LABEL: aag 3 1 1 1 1
// CHECK-NEXT: 2
// CHECK-NEXT: 4 6
// CHECK-NEXT: 4
// CHECK-NEXT: 6 4 2
hw.module @register(in %clk: !seq.clock, in %input: i1, out output: i1) {
  %and_result = synth.aig.and_inv %reg, %input : i1
  %reg = seq.compreg %and_result, %clk : i1
  hw.output %reg : i1
}

// -----

// Test multiple AND gates in sequence
// CHECK-LABEL: aag 5 3 0 1 2
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 10
// CHECK-NEXT: 8 2 4
// CHECK-NEXT: 10 6 8
hw.module @chain_ands(in %a: i1, in %b: i1, in %c: i1, out result: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %c, %0 : i1
  hw.output %1 : i1
}

// -----

// Test multiple outputs
// CHECK-LABEL: aag 4 2 0 2 2
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 8
// CHECK-NEXT: 6 2 4
// CHECK-NEXT: 8 6 4
hw.module @multiple_outputs(in %a: i1, in %b: i1, out out1: i1, out out2: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %0, %b : i1
  hw.output %0, %1 : i1, i1
}

// -----

// Test replication handling
// CHECK-LABEL: aag 3 3 0 2 0
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 6
// CHECK-NEXT: 4
hw.module @replicate_test(in %input: i3, out result1: i1, out result2: i1) {
  %replicate = comb.replicate %input : (i3) -> i6
  %extract1 = comb.extract %replicate from 5 : (i6) -> i1
  %extract2 = comb.extract %replicate from 4 : (i6) -> i1
  hw.output %extract1, %extract2 : i1, i1
} 

// -----
// Test constant propagation scenarios
// CHECK-LABEL: aag 2 1 0 2 1
// CHECK-NEXT: 2
// CHECK-NEXT: 2
// CHECK-NEXT: 1
// CHECK-NEXT: 4 2 1
hw.module @constant_propagation(in %input: i1, out always_input: i1, out always_true: i1) {
  %true = hw.constant true
  %and_with_true = synth.aig.and_inv %input, %true : i1
  hw.output %input, %true : i1, i1
}

// -----

// Test multi-bit inputs and outputs (should be flattened to individual bits)
// CHECK-LABEL: aag 4 4 0 4 0
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 8
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 8
hw.module @multibit_io(in %a: i2, in %b: i2, out x: i2, out y: i2) {
  hw.output %a, %b : i2, i2
}

// -----

// Test concatenation of single bits
// CHECK-LABEL: aag 2 2 0 2 0
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 4
// CHECK-NEXT: 2
hw.module @concat_bits(in %a: i1, in %b: i1, out result: i2) {
  %concat = comb.concat %a, %b : i1, i1
  hw.output %concat : i2
}

// -----

// Test extraction from multi-bit value
// CHECK-LABEL: aag 2 2 0 2 0
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 2
// CHECK-NEXT: 4
hw.module @extract_bits(in %input: i2, out bit0: i1, out bit1: i1) {
  %bit0 = comb.extract %input from 0 : (i2) -> i1
  %bit1 = comb.extract %input from 1 : (i2) -> i1
  hw.output %bit0, %bit1 : i1, i1
}

// -----

// Test complex concatenation with mixed widths
// CHECK-LABEL: aag 3 3 0 4 0
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 2
hw.module @complex_concat(in %a: i1, in %b: i2, out result: i4) {
  %concat = comb.concat %a, %b, %a : i1, i2, i1
  hw.output %concat : i4
}

// -----

// Test AND operations on multi-bit values (bit-wise)
// CHECK-LABEL: aag 6 4 0 2 2
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 8
// CHECK-NEXT: 10
// CHECK-NEXT: 12
// CHECK-NEXT: 10 2 6
// CHECK-NEXT: 12 4 8
hw.module @multibit_and(in %a: i2, in %b: i2, out result: i2) {
  %0 = synth.aig.and_inv %a, %b : i2
  hw.output %0 : i2
}

// -----

// Test registers with multi-bit values
// CHECK-LABEL: aag 6 2 2 2 2
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6 10
// CHECK-NEXT: 8 12
// CHECK-NEXT: 6
// CHECK-NEXT: 8
// CHECK-NEXT: 10 2 6
// CHECK-NEXT: 12 4 8
hw.module @multibit_register(in %clk: !seq.clock, in %input: i2, out output: i2) {
  %and_result = synth.aig.and_inv %input, %reg : i2
  %reg = seq.compreg %and_result, %clk : i2
  hw.output %reg : i2
}

// -----

// Test constants with multiple bits
// CHECK-LABEL: aag 0 0 0 3 0
// CHECK-NEXT: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 0
hw.module @multibit_constants(out low: i1, out high: i1, out zero: i1) {
  %c1_i2 = hw.constant 0 : i2
  %c2_i2 = hw.constant 3 : i2
  %low_bit = comb.extract %c1_i2 from 0 : (i2) -> i1
  %high_bit = comb.extract %c2_i2 from 0 : (i2) -> i1
  %zero_bit = comb.extract %c1_i2 from 1 : (i2) -> i1
  hw.output %low_bit, %high_bit, %zero_bit : i1, i1, i1
}

// -----

// Test comments
// CHECK-LABEL: aag 0 0 0 0 0
// CHECK-NEXT: c
// CHECK-NEXT: Generated by CIRCT
// CHECK-NEXT: module: comment
hw.module @comment() {
}
