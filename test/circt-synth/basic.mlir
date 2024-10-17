// RUN: circt-synth %s | FileCheck %s

// CHECK-LABEL: @and
hw.module @and(in %a: i1, in %b: i1, out and: i1) {
  %0 = comb.and %a, %b : i1
  // CHECK-NEXT: %0 = comb.truth_table %a, %b -> [false, false, false, true]
  hw.output %0 : i1
}

// CHECK-LABEL: @or
hw.module @or(in %a: i1, in %b: i1, out or: i1) {
  %0 = comb.or %a, %b : i1
  // CHECK-NEXT: %0 = comb.truth_table %a, %b -> [false, true, true, true]
  hw.output %0 : i1
}

// CHECK-LABEL: @xor
hw.module @xor(in %a: i1, in %b: i1, out xor: i1) {
  %0 = comb.xor %a, %b : i1
  // CHECK-NEXT: %0 = comb.truth_table %a, %b -> [false, true, true, false]
  hw.output %0 : i1
}

// CHECK-LABEL: @multibit
hw.module @multibit(in %a: i2, in %b: i2, out and: i2) {
  %0 = comb.and %a, %b : i2
  // CHCK-NEXT: %0 = comb.extract %a from 0 : (i2) -> i1
  // CHCK-NEXT: %1 = comb.extract %b from 0 : (i2) -> i1
  // CHCK-NEXT: %2 = comb.truth_table %0, %1 -> [false, false, false, true]
  // CHCK-NEXT: %3 = comb.extract %a from 1 : (i2) -> i1
  // CHCK-NEXT: %4 = comb.extract %b from 1 : (i2) -> i1
  // CHCK-NEXT: %5 = comb.truth_table %3, %4 -> [false, false, false, true]
  // CHCK-NEXT: %6 = comb.concat %2, %5 : i1, i1
  hw.output %0 : i2
}

// CHECK-LABEL: @variadic
hw.module @variadic(in %a: i1, in %b: i1, in %c: i1,
                    in %d: i1, in %e: i1, in %f: i1, out and6: i1) {
  %0 = comb.and %a, %b, %c, %d, %e, %f : i1
  // CHECK-NEXT: %0 = comb.truth_table %a, %b, %c, %d, %e, %f -> [
  // CHECK-COUNT-63: false
  // CHECK-SAME: true
  // CHECK-SAME: ]
  hw.output %0 : i1
}