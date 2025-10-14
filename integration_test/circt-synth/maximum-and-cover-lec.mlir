// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --convert-synth-to-comb -o %t1.mlir
// RUN: circt-opt %s --synth-maximum-and-cover --convert-synth-to-comb -o %t2.mlir

// RUN: circt-lec %t1.mlir %t2.mlir -c1=MaxCover1 -c2=MaxCover1 --shared-libs=%libz3 | FileCheck %s --check-prefix=MAX_COVER_1
// MAX_COVER_1: c1 == c2
hw.module @MaxCover1(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out o1: i1, out o2: i1, out o3: i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %0, %c : i1
  %2 = synth.aig.and_inv %b, %c : i1
  %3 = synth.aig.and_inv %2, %d : i1

  %4 = synth.aig.and_inv %c, %d : i1
  %5 = synth.aig.and_inv %b, %4 : i1
  %6 = synth.aig.and_inv %a, %5 : i1

  hw.output %1, %3, %6 : i1, i1, i1
}

// RUN: circt-lec %t1.mlir %t2.mlir -c1=MaxCover2 -c2=MaxCover2 --shared-libs=%libz3 | FileCheck %s --check-prefix=MAX_COVER_2
// MAX_COVER_2: c1 == c2
hw.module @MaxCover2(in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1, in %g: i1, out o1: i1) {
  %1 = synth.aig.and_inv %a, not %b : i1
  %2 = synth.aig.and_inv %d, not %e : i1
  %3 = synth.aig.and_inv not %2, %f : i1
  %4 = synth.aig.and_inv not %c, %3 : i1
  %5 = synth.aig.and_inv %1, not %4 : i1
  %6 = synth.aig.and_inv %5, %g : i1

  hw.output %6 : i1
}
