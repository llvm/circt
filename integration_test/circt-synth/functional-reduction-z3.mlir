// REQUIRES: z3-integration, libz3
// RUN: circt-opt %s -pass-pipeline='builtin.module(hw.module(synth-functional-reduction{num-random-patterns=64 sat-solver=z3}))' -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 functional_reduction_sat --c2 functional_reduction_sat | FileCheck %s --check-prefix=BASIC
// BASIC: c1 == c2
// SAT should prove that AND(AND(a, not b), AND(c, not d)) is equivalent to
// AND(a, not b, c, not d), and the pass should materialize that with a choice.
// CHECK-LABEL: hw.module @functional_reduction_sat
hw.module @functional_reduction_sat(in %a: i1, in %b: i1, in %c: i1, in %d: i1,
                                    out out1: i1, out out2: i1, out out3: i1) {
  // CHECK: %[[AB:.+]] = synth.aig.and_inv %a, not %b : i1
  // CHECK-NEXT: %[[CD:.+]] = synth.aig.and_inv %c, not %d : i1
  // CHECK-NEXT: %[[TREE:.+]] = synth.aig.and_inv %[[AB]], %[[CD]] : i1
  // CHECK-NEXT: %[[FLAT:.+]] = synth.aig.and_inv %a, not %b, %c, not %d : i1
  // CHECK-NEXT: %[[CHOICE:.+]] = synth.choice %[[TREE]], %[[FLAT]]
  // CHECK-NEXT: %[[DIFF:.+]] = synth.aig.and_inv %a, not %b, not %c, not %d : i1
  // CHECK-NEXT: hw.output %[[CHOICE]], %[[CHOICE]], %[[DIFF]] : i1, i1, i1
  %0 = synth.aig.and_inv %a, not %b : i1
  %1 = synth.aig.and_inv %c, not %d : i1
  %2 = synth.aig.and_inv %0, %1 : i1
  %3 = synth.aig.and_inv %a, not %b, %c, not %d : i1
  %4 = synth.aig.and_inv %a, not %b, not %c, not %d : i1
  hw.output %2, %3, %4 : i1, i1, i1
}

// SAT should also prove equivalence across the newly supported comb
// nodes.
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 functional_reduction_supported_ops_sat --c2 functional_reduction_supported_ops_sat | FileCheck %s --check-prefix=MIXED
// MIXED: c1 == c2
// CHECK-LABEL: hw.module @functional_reduction_supported_ops_sat
hw.module @functional_reduction_supported_ops_sat(
    in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1,
    out out0: i1, out out1: i1, out out2: i1, out out3: i1,
    out out4: i1, out out5: i1) {
  // CHECK: hw.output %[[ORCHOICE:.+]], %[[ORCHOICE]],
  // CHECK-SAME:      %[[ANDCHOICE:.+]], %[[ANDCHOICE]],
  // CHECK-SAME:      %[[XORCHOICE:.+]], %[[XORCHOICE]]
  %false = hw.constant false
  %0 = comb.or %a, %b, %c : i1
  %1 = comb.or %c, %b, %a : i1
  %2 = comb.and %a, %b : i1
  %3 = comb.and %b, %a : i1
  %4 = comb.xor %a, %b : i1
  %5 = comb.xor %b, %a : i1
  hw.output %0, %1, %2, %3, %4, %5 : i1, i1, i1, i1, i1, i1
}

// SAT should satisfy inversion equivalences
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 functional_reduction_inversion_equiv_sat --c2 functional_reduction_inversion_equiv_sat | FileCheck %s --check-prefix=INVERSION
// INVERSION: c1 == c2
// CHECK-LABEL: hw.module @functional_reduction_inversion_equiv_sat
hw.module @functional_reduction_inversion_equiv_sat(in %a: i1, in %b: i1,
                                                    out out0: i1, out out1: i1) {
  // CHECK: %[[AND:.+]] = synth.aig.and_inv not %a, not %b : i1
  // CHECK: %[[OR:.+]] = comb.or %a, %b : i1
  // CHECK: %[[NOTMEMBER:.+]] = synth.aig.and_inv not %[[OR]]
  // CHECK: %[[CHOICE:.+]] = synth.choice %[[AND]], %[[NOTMEMBER]]
  // CHECK: %[[CHOICENOT:.+]] = synth.aig.and_inv not %[[CHOICE]]
  // CHECK: hw.output %[[CHOICE]], %[[CHOICENOT]]
  %0 = synth.aig.and_inv not %a, not %b : i1
  %1 = comb.or %a, %b : i1
  hw.output %0, %1 : i1, i1
}

// SAT should also prove equivalence between synth.xor_inv and an AND/OR
// implementation of XOR.
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 functional_reduction_xor_inv_sat --c2 functional_reduction_xor_inv_sat | FileCheck %s --check-prefix=XOR-INV
// XOR-INV: c1 == c2
// CHECK-LABEL: hw.module @functional_reduction_xor_inv_sat
hw.module @functional_reduction_xor_inv_sat(in %a: i1, in %b: i1,
                                            out out0: i1, out out1: i1) {
  // CHECK: hw.output %[[CHOICE:.+]], %[[CHOICE]] : i1, i1
  %0 = synth.xor_inv %a, %b : i1
  %1 = synth.aig.and_inv not %a, %b : i1
  %2 = synth.aig.and_inv %a, not %b : i1
  %3 = synth.aig.and_inv not %1, not %2 : i1
  %4 = synth.aig.and_inv not %3 : i1
  hw.output %0, %4 : i1, i1
}
