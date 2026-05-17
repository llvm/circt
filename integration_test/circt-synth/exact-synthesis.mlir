// REQUIRES: z3-integration, circt-lec-jit
// RUN: circt-opt %s --lower-comb -o %t.lowered.mlir

// and2 exact synthesis.
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.aig.and_inv:2 sat-solver=z3})' -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefixes=CHECK,AND2
// RUN: circt-lec %t.lowered.mlir %t.mlir -c1=test -c2=test --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC

// and3 exact synthesis.
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.aig.and_inv:3 sat-solver=z3})' -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefixes=CHECK,AND3
// RUN: circt-lec %t.lowered.mlir %t.mlir -c1=test -c2=test --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC

// dot exact synthesis.
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.dot:3 sat-solver=z3})' -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefixes=CHECK,DOT
// RUN: circt-lec %t.lowered.mlir %t.mlir -c1=test -c2=test --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC

// xag exact synthesis.
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.xor_inv:2 allowed-ops=synth.aig.and_inv:2 sat-solver=z3})' -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefixes=CHECK,XAG
// RUN: circt-lec %t.lowered.mlir %t.mlir -c1=test -c2=test --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC

// Test exact synthesis of some simple functions.
// CHECK-LABEL: hw.module @test
// AND2: synth.aig.and_inv {{.*}}, {{.*}} : i1
// AND3: synth.aig.and_inv {{.*}}, {{.*}}, {{.*}} : i1
// DOT: synth.dot {{.*}}, {{.*}}, {{.*}} : i1
// XAG-DAG: synth.xor_inv {{.*}}, {{.*}} : i1
// XAG-DAG: synth.aig.and_inv {{.*}}, {{.*}} : i1
// CHECK-NOT: comb.truth_table
// CHECK: hw.output
// LEC: c1 == c2

hw.module @test(in %a : i1, in %b : i1, in %c : i1,
                out y : i1) {
  %0 = comb.truth_table %a, %b, %c -> [false, true, false, true,
                                        true, false, false, false]
  hw.output %0 : i1
}

// Test direct synthesis (no gate needs to be synthesized).
// CHECK-LABEL: hw.module @constant_false
// CHECK: %[[FALSE:.+]] = hw.constant false
// CHECK-NEXT: hw.output %[[FALSE]] : i1

// CHECK-LABEL: hw.module @constant_true
// CHECK: %[[TRUE:.+]] = hw.constant true
// CHECK-NEXT: hw.output %[[TRUE]] : i1

// CHECK-LABEL: hw.module @projection
// CHECK-NEXT: hw.output %a : i1

// CHECK-LABEL: hw.module @inverted_projection
// CHECK: %[[NOT_A:.+]] = synth.aig.and_inv not %a : i1
// CHECK-NEXT: hw.output %[[NOT_A]] : i1

hw.module @constant_false(in %a : i1, out y : i1) {
  %0 = comb.truth_table %a -> [false, false]
  hw.output %0 : i1
}

hw.module @constant_true(in %a : i1, out y : i1) {
  %0 = comb.truth_table %a -> [true, true]
  hw.output %0 : i1
}

hw.module @projection(in %a : i1, in %b : i1, out y : i1) {
  %0 = comb.truth_table %a, %b -> [false, false, true, true]
  hw.output %0 : i1
}

hw.module @inverted_projection(in %a : i1, in %b : i1, out y : i1) {
  %0 = comb.truth_table %a, %b -> [true, true, false, false]
  hw.output %0 : i1
}
