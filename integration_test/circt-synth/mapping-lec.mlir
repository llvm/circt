// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-synth %s -o %t1.mlir
// RUN: cat %t1.mlir | FileCheck %s
// RUN: circt-lec %t1.mlir %s -c1=mul -c2=mul --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MUL_TECHMAP

// COMB_MUL_TECHMAP: c1 == c2

// RUN: circt-synth %s -o %t.lut.mlir --top mul --lower-to-k-lut 6
// RUN: cat %t.lut.mlir | FileCheck %s --check-prefix=LUT
// RUN: circt-opt -lower-comb %t.lut.mlir -o %t2.mlir
// RUN: circt-lec %t2.mlir %s -c1=mul -c2=mul --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MUL_LUT

// COMB_MUL_LUT: c1 == c2

// Set delay for binary and inv op to 5 so that others will be prioritized
hw.module @and_inv(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[5], [5]]}} {
    %0 = synth.aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_inv_n(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[5], [5]]}} {
    %0 = synth.aig.and_inv not %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_inv_nn(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[5], [5]]}} {
    %0 = synth.aig.and_inv not %a, not %b : i1
    hw.output %0 : i1
}

hw.module @nand_nand(in %a : i1, in %b : i1, in %c : i1, in %d: i1, out result : i1) attributes {hw.techlib.info = {area = 3.0 : f64, delay = [[1], [1], [1], [1]]}} {
    %0 = synth.aig.and_inv %a, %b : i1
    %1 = synth.aig.and_inv %c, %d : i1
    %2 = synth.aig.and_inv not %0, not %1 : i1
    hw.output %2 : i1
}

hw.module @some(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
    %0 = synth.aig.and_inv not %a, not %b : i1
    %1 = synth.aig.and_inv %a, %b : i1
    %2 = synth.aig.and_inv not %0, not %1 : i1
    hw.output %2 : i1
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
