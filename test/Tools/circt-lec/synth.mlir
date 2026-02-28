// RUN: circt-lec %s --c1 foo1 --c2 foo2 -emit-mlir | FileCheck %s

// CHECK-LABEL: smt.solver()
// CHECK: smt.bv.and
// CHECK: smt.bv.or
hw.module @foo1(in %a : i8, in %b : i8, in %c: i8, out d : i8) {
  %0 = synth.aig.and_inv %a, %b, %c: i8
  hw.output %0 : i8
}

hw.module @foo2(in %a : i8, in %b : i8, in %c: i8, out d : i8) {
  %0 = synth.mig.maj_inv %a, %b, %c: i8
  hw.output %0 : i8
}
