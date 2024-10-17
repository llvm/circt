// RUN: circt-synth %s --lut-size=2 | FileCheck %s --check-prefix=LUT-2
// RUN: circt-synth %s --lut-size=4 | FileCheck %s --check-prefix=LUT-4
// RUN: circt-synth %s --lut-size=6 | FileCheck %s --check-prefix=LUT-6

// LUT-2-LABEL: @variadic
// LUT-4-LABEL: @variadic
// LUT-6-LABEL: @variadic
hw.module @variadic(in %a: i1, in %b: i1, in %c: i1,
                    in %d: i1, in %e: i1, in %f: i1, out and6: i1) {
  // LUT-2-NEXT: %0 = comb.truth_table %b, %c -> [false, false, false, true]
  // LUT-2-NEXT: %1 = comb.truth_table %a, %0 -> [false, false, false, true]
  // LUT-2-NEXT: %2 = comb.truth_table %e, %f -> [false, false, false, true]
  // LUT-2-NEXT: %3 = comb.truth_table %d, %2 -> [false, false, false, true]
  // LUT-2-NEXT: %4 = comb.truth_table %1, %3 -> [false, false, false, true]
  // LUT-2-NEXT: hw.output %4
  // LUT-4-NEXT: %0 = comb.truth_table %a, %b, %c -> [false, false, false, false, false, false, false, true] 
  // LUT-4-NEXT: %1 = comb.truth_table %0, %d, %e, %f -> [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true] 
  // LUT-4-NEXT: hw.output %1
  // LUT-6-NEXT: %0 = comb.truth_table 
  // LUT-6-NEXT: hw.output %0
  %0 = comb.and %a, %b, %c, %d, %e, %f : i1

  hw.output %0 : i1
}
