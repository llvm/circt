// REQUIRES: z3

// RUN: circt-synth %s -o %t1.mlir
// RUN: circt-opt %t1.mlir --hw-flatten-modules=hw-inline-public -o %t1.inline.mlir
// RUN: circt-lec.sh %t1.inline.mlir %s -c1=mul -c2=mul
// RUN: circt-lec.sh %t1.inline.mlir %s -c1=dot_test -c2=dot_test

// RUN: circt-synth %s -o %t.lut.mlir --top mul --lower-to-k-lut 6
// RUN: circt-opt -lower-comb %t.lut.mlir -o %t2.mlir
// RUN: circt-lec.sh %t2.mlir %s -c1=mul -c2=mul


// Set delay for binary and inv op to 5 so that others will be prioritized
hw.module @and_inv(in %a : i1, in %b : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 5, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "b", 5, 0, #synth.polarity<positive>>], input_caps = {}>} {
    %0 = synth.aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_inv_n(in %a : i1, in %b : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 5, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "b", 5, 0, #synth.polarity<positive>>], input_caps = {}>} {
    %0 = synth.aig.and_inv not %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_inv_nn(in %a : i1, in %b : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 5, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "b", 5, 0, #synth.polarity<positive>>], input_caps = {}>} {
    %0 = synth.aig.and_inv not %a, not %b : i1
    hw.output %0 : i1
}

hw.module @nand_nand(in %a : i1, in %b : i1, in %c : i1, in %d: i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 3.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "b", 1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "c", 1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "d", 1, 0, #synth.polarity<positive>>], input_caps = {}>} {
    %0 = synth.aig.and_inv %a, %b : i1
    %1 = synth.aig.and_inv %c, %d : i1
    %2 = synth.aig.and_inv not %0, not %1 : i1
    hw.output %2 : i1
}

hw.module @some(in %a : i1, in %b : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "b", 1, 0, #synth.polarity<positive>>], input_caps = {}>} {
    %0 = synth.aig.and_inv not %a, not %b : i1
    %1 = synth.aig.and_inv %a, %b : i1
    %2 = synth.aig.and_inv not %0, not %1 : i1
    hw.output %2 : i1
}

hw.module @dot_lib(in %x : i1, in %y : i1, in %z : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "x", 1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "y", 1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "z", 1, 0, #synth.polarity<positive>>], input_caps = {}>} {
    %0 = synth.dot %z, not %x, not %y : i1
    hw.output %0 : i1
}

hw.module @dot_test(in %x : i1, in %y : i1, in %z : i1, out result : i1) {
    %0 = synth.dot %x, not %y, not %z : i1
    hw.output %0 : i1
}

// Make sure @mul is mapped to modules above.
// CHECK-LABEL: hw.module @mul
// CHECK-NOT: synth.aig.and_inv
// CHECK-NOT: comb.and
// CHECK-NOT: comb.xor
// CHECK-DAG: hw.instance {{".+"}} @and_inv
// CHECK-DAG: hw.instance {{".+"}} @some
// FIXME: To map @nand_nand it's necessary to implement supergate generation.
// CHECK-NOT: hw.instance {{".+"}} @nand_nand
// CHECK-DAG: hw.instance {{".+"}} @and_inv_n
// CHECK-DAG: hw.instance {{".+"}} @and_inv_nn
// LUT: hw.module @mul
// LUT: comb.truth_table
// LUT-NOT: synth.aig.and_inv
// LUT-NOT: comb.and
// LUT-NOT: comb.xor
// LUT-NOT: hw.instance
hw.module @mul(in %arg0: i4, in %arg1: i4, out add: i4) {
  %0 = comb.mul %arg0, %arg1 : i4
  hw.output %0 : i4
}
