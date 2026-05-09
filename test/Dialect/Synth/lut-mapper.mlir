// FIXME: max-cuts-per-root=20 is due to a lack of non-minimal cut filtering.
// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(synth-generic-lut-mapper{test=true max-cuts-per-root=20}))' %s | FileCheck %s --check-prefixes CHECK,LUT
// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(synth-generic-lut-mapper{test=true max-lut-size=2}))' %s | FileCheck %s --check-prefixes CHECK,LUT2

// CHECK:      %[[B_0:.+]] = comb.extract %b from 0 : (i2) -> i1
// CHECK-NEXT: %[[B_1:.+]] = comb.extract %b from 1 : (i2) -> i1
// CHECK-NEXT: %[[A_0:.+]] = comb.extract %a from 0 : (i2) -> i1
// CHECK-NEXT: %[[A_1:.+]] = comb.extract %a from 1 : (i2) -> i1

// LUT-NEXT:   %[[C_0:.+]] = comb.truth_table %[[A_0]], %[[B_0]] -> [false, true, true, false]
// LUT-SAME:   test.arrival_times = [1]
// LUT-NEXT:   %[[C_1:.+]] = comb.truth_table %[[A_1]], %[[A_0]], %[[B_1]], %[[B_0]] -> [false, false, true, true, false, true, true, false, true, true, false, false, true, false, false, true]
// LUT-SAME:   test.arrival_times = [1]
// LUT-NEXT:   %[[C_2:.+]] = comb.concat %[[C_1]], %[[C_0]] : i1, i1
// LUT-NEXT:   hw.output %[[C_2]] : i2

// LUT2:       %[[C_0:.+]] = comb.truth_table %[[A_0]], %[[B_0]] -> [false, false, false, true]
// LUT2-SAME:   test.arrival_times = [1]
// LUT2-NEXT:  %[[C_1:.+]] = comb.truth_table %[[A_0]], %[[B_0]] -> [false, true, true, false]
// LUT2-SAME:  test.arrival_times = [1]
// LUT2-NEXT:  %[[C_2:.+]] = comb.truth_table %[[C_0]], %[[A_1]] -> [false, true, true, false]
// LUT2-SAME:  test.arrival_times = [2]
// LUT2-NEXT:  %[[C_3:.+]] = comb.truth_table %[[C_2]], %[[B_1]] -> [false, true, true, false]
// LUT2-SAME:  test.arrival_times = [3]
// LUT2-NEXT:  %[[C_4:.+]] = comb.concat %[[C_3]], %[[C_1]] : i1, i1
// LUT2-NEXT:  hw.output %[[C_4]] : i2
hw.module @add(in %a : i2, in %b : i2, out result : i2) {
  %0 = comb.extract %b from 0 : (i2) -> i1 
  %1 = comb.extract %b from 1 : (i2) -> i1
  %2 = comb.extract %a from 0 : (i2) -> i1
  %3 = comb.extract %a from 1 : (i2) -> i1
  %4 = synth.aig.and_inv not %0, not %2 : i1
  %5 = synth.aig.and_inv %0, %2 : i1
  %6 = synth.aig.and_inv not %4, not %5 : i1
  %7 = synth.aig.and_inv not %3, not %5 : i1
  %8 = synth.aig.and_inv %3, %5 : i1
  %9 = synth.aig.and_inv not %7, not %8 : i1
  %10 = synth.aig.and_inv not %1, not %9 : i1
  %11 = synth.aig.and_inv %1, %9 : i1
  %12 = synth.aig.and_inv not %10, not %11 : i1
  %13 = comb.concat %12, %6 : i1, i1
  hw.output %13 : i2
}

// CHECK-LABEL: hw.module @choice_slow_branch
// LUT-NEXT:   %[[OUT:.+]] = comb.truth_table %c, %b, %a -> [false, false, false, false, false, false, false, true]
// LUT-SAME:   test.arrival_times = [1]
// LUT-NEXT:   hw.output %[[OUT]] : i1
// LUT2-NEXT:  %[[AB:.+]] = comb.truth_table %b, %a -> [false, false, false, true]
// LUT2-SAME:  test.arrival_times = [1]
// LUT2-NEXT:  %[[OUT:.+]] = comb.truth_table %[[AB]], %c -> [false, false, false, true]
// LUT2-SAME:  test.arrival_times = [2]
// LUT2-NEXT:  hw.output %[[OUT]] : i1
hw.module @choice_slow_branch(in %a : i1, in %b : i1, in %c : i1,
                              out y : i1) {
  %fast0 = synth.aig.and_inv %a, %b : i1
  %fast = synth.aig.and_inv %fast0, %c : i1

  %na = synth.aig.and_inv not %a : i1
  %a2 = synth.aig.and_inv not %na : i1
  %nb = synth.aig.and_inv not %b : i1
  %b2 = synth.aig.and_inv not %nb : i1
  %nc = synth.aig.and_inv not %c : i1
  %c2 = synth.aig.and_inv not %nc : i1
  %slow0 = synth.aig.and_inv %a2, %b2 : i1
  %slow = synth.aig.and_inv %slow0, %c2 : i1

  %choice = synth.choice %slow, %fast : i1
  hw.output %choice : i1
}
