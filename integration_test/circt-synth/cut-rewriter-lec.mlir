// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// Perform translation validaton of LUT mapping in order to verify the truth
// table computation in the cut rewriter.
// RUN: circt-opt -synth-generic-lut-mapper -lower-comb %s -o %t.mlir
// RUN: circt-lec %t.mlir %s -c1=test -c2=test --shared-libs=%libz3 | FileCheck %s

// CHECK: c1 == c2

hw.module @test(in %a : i1, in %b : i1, in %c : i1, out result : i1) {
    %0 = synth.aig.and_inv %a, not %b : i1
    %1 = comb.xor %c, %0 : i1
    hw.output %1 : i1
}
