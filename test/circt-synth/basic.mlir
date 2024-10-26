// RUN: circt-synth %s | FileCheck %s

// CHECK-LABEL: @and
hw.module @and(in %a: i1, in %b: i1, out and: i1) {
  %0 = comb.and %a, %b : i1
  // CHECK-NEXT: %[[RESULT:.+]] = comb.truth_table %a, %b -> [false, false, false, true]
  // CHECK-NEXT: hw.output %[[RESULT:.+]] : i1
  hw.output %0 : i1
}

// CHECK-LABEL: @or
hw.module @or(in %a: i1, in %b: i1, out or: i1) {
  %0 = comb.or %a, %b : i1
  // CHECK-NEXT: %[[RESULT:.+]] = comb.truth_table %a, %b -> [false, true, true, true]
  // CHECK-NEXT: hw.output %[[RESULT:.+]] : i1
  hw.output %0 : i1
}

// CHECK-LABEL: @xor
hw.module @xor(in %a: i1, in %b: i1, out xor: i1) {
  %0 = comb.xor %a, %b : i1
  // CHECK-NEXT: %[[RESULT:.+]] = comb.truth_table %a, %b -> [false, true, true, false]
  // CHECK-NEXT: hw.output %[[RESULT:.+]] : i1
  hw.output %0 : i1
}

// CHECK-LABEL: @multibit
hw.module @multibit(in %a: i2, in %b: i2, out and: i2) {
  %0 = comb.and %a, %b : i2
  // CHECK-NEXT: %[[EXTRACT0:.+]] = comb.extract %a from 0 : (i2) -> i1
  // CHECK-NEXT: %[[EXTRACT1:.+]] = comb.extract %b from 0 : (i2) -> i1
  // CHECK-NEXT: %[[LUT0:.+]] = comb.truth_table %[[EXTRACT0]], %[[EXTRACT1]] -> [false, false, false, true]
  // CHECK-NEXT: %[[EXTRACT2:.+]] = comb.extract %a from 1 : (i2) -> i1
  // CHECK-NEXT: %[[EXTRACT3:.+]] = comb.extract %b from 1 : (i2) -> i1
  // CHECK-NEXT: %[[LUT1:.+]] = comb.truth_table %[[EXTRACT2]], %[[EXTRACT3]] -> [false, false, false, true]
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %[[LUT0]], %[[LUT1]] : i1, i1
  hw.output %0 : i2
}

// CHECK-LABEL: @variadic
hw.module @variadic(in %a: i1, in %b: i1, in %c: i1,
                    in %d: i1, in %e: i1, in %f: i1, out and6: i1) {
  %0 = comb.and %a, %b, %c, %d, %e, %f : i1
  // CHECK-NEXT: %[[RESULT:.+]] = comb.truth_table %a, %b, %c, %d, %e, %f -> [
  // CHECK-COUNT-63: false
  // CHECK-SAME: true
  // CHECK-SAME: ]
  // CHECK-NEXT: hw.output %[[RESULT:.+]] : i1
  hw.output %0 : i1
}
