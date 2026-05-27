// RUN: circt-lec %s --c1 table --c2 xor --emit-mlir | FileCheck %s

// CHECK-LABEL: smt.solver()
// CHECK-NOT: comb.truth_table
// CHECK: smt.ite
hw.module @table(in %a : i1, in %b : i1, out result : i1) {
  %0 = comb.truth_table %a, %b -> [false, true, true, false]
  hw.output %0 : i1
}

hw.module @xor(in %a : i1, in %b : i1, out result : i1) {
  %0 = comb.xor %a, %b : i1
  hw.output %0 : i1
}
