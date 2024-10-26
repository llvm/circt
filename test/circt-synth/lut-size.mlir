// RUN: circt-synth %s --lut-size=2 | FileCheck %s --check-prefix=LUT-2
// RUN: circt-synth %s --lut-size=4 | FileCheck %s --check-prefix=LUT-4
// RUN: circt-synth %s --lut-size=6 | FileCheck %s --check-prefix=LUT-6

// LUT-2-LABEL: @variadic
// LUT-4-LABEL: @variadic
// LUT-6-LABEL: @variadic
hw.module @variadic(in %a: i1, in %b: i1, in %c: i1,
                    in %d: i1, in %e: i1, in %f: i1, out and6: i1) {
  // LUT-2-NEXT: %[[LUT0:.+]] = comb.truth_table %b, %c -> [false, false, false, true]
  // LUT-2-NEXT: %[[LUT1:.+]] = comb.truth_table %a, %[[LUT0]] -> [false, false, false, true]
  // LUT-2-NEXT: %[[LUT2:.+]] = comb.truth_table %e, %f -> [false, false, false, true]
  // LUT-2-NEXT: %[[LUT3:.+]] = comb.truth_table %d, %[[LUT2]] -> [false, false, false, true]
  // LUT-2-NEXT: %[[LUT4:.+]] = comb.truth_table %[[LUT1]], %[[LUT3]] -> [false, false, false, true]
  // LUT-2-NEXT: hw.output %[[LUT4]]
  // LUT-4-NEXT: %[[LUT0:.+]] = comb.truth_table %a, %b, %c -> [false, false, false, false, false, false, false, true]
  // LUT-4-NEXT: %[[LUT1:.+]] = comb.truth_table %[[LUT0]], %d, %e, %f -> [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true]
  // LUT-4-NEXT: hw.output %[[LUT1]]
  // LUT-6-NEXT: %[[LUT0:.+]] = comb.truth_table
  // LUT-6-NEXT: hw.output %[[LUT0]]
  %0 = comb.and %a, %b, %c, %d, %e, %f : i1

  hw.output %0 : i1
}
