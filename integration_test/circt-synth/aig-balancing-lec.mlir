// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --cse --convert-aig-to-comb -o %t1.mlir
// RUN: circt-opt %s --maximum-and-cover --aig-balance-variadic --cse --convert-aig-to-comb -o %t2.mlir

// RUN: circt-lec %t.mlir %s -c1=aig -c2=aig --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_AIG
// COMB_AIG: c1 == c2
hw.module @aig(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out o1: i1, out o2: i1, out o3: i1) {
  %0 = aig.and_inv %a, %b : i1
  %1 = aig.and_inv %0, %c : i1
  %2 = aig.and_inv %b, %c : i1
  %3 = aig.and_inv %2, %d : i1

  %4 = aig.and_inv %c, %d : i1
  %5 = aig.and_inv %b, %4 : i1
  %6 = aig.and_inv %a, %5 : i1

  hw.output %1, %3, %6 : i1, i1, i1
}