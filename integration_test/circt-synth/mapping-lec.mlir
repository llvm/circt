// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-synth %s -o %t1.mlir -convert-to-comb --top mul
// RUN: cat %t1.mlir | FileCheck %s
// RUN: circt-opt %s --convert-aig-to-comb -o %t2.mlir
// RUN: circt-lec %t1.mlir %t2.mlir -c1=mul -c2=mul --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MUL_TECHMAP

// COMB_MUL_TECHMAP: c1 == c2

// Set delay for binary and inv op to 5 so that others will be prioritized
hw.module @and_inv(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[5], [5]]}} {
    %0 = aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_inv_n(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[5], [5]]}} {
    %0 = aig.and_inv not %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_inv_nn(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[5], [5]]}} {
    %0 = aig.and_inv not %a, not %b : i1
    hw.output %0 : i1
}

hw.module @nand_nand(in %a : i1, in %b : i1, in %c : i1, in %d: i1, out result : i1) attributes {hw.techlib.info = {area = 3.0 : f64, delay = [[1], [1], [1], [1]]}} {
    %0 = aig.and_inv %a, %b : i1
    %1 = aig.and_inv %c, %d : i1
    %2 = aig.and_inv not %0, not %1 : i1
    hw.output %2 : i1
}

hw.module @some(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
    %0 = aig.and_inv not %a, not %b : i1
    %1 = aig.and_inv %a, %b : i1
    %2 = aig.and_inv not %0, not %1 : i1
    hw.output %2 : i1
}

// Make sure @mul is mapped to modules above.
// CHECK-LABEL: hw.module @mul
// CHECK-NOT: aig.and_inv
// CHECK-NOT: comb.and
// CHECK-NOT: comb.xor
// CHECK-DAG: hw.instance {{".+"}} @and_inv
// CHECK-DAG: hw.instance {{".+"}} @some
// CHECK-DAG: hw.instance {{".+"}} @nand_nand
// CHECK-DAG: hw.instance {{".+"}} @and_inv_n
// CHECK-DAG: hw.instance {{".+"}} @and_inv_nn
hw.module @mul(in %arg0: i4, in %arg1: i4, out add: i4) {
  %0 = comb.mul %arg0, %arg1 : i4
  hw.output %0 : i4
}
