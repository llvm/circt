// RUN: circt-opt %s --comb-overflow-annotating | FileCheck %s

// CHECK-LABEL: @basic_add
hw.module @basic_add(in %a : i4, in %b : i4, out add4 : i4, out add5 : i5) {
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: comb.add %a, %b {comb.nuw = true} : i4
  // CHECK-NEXT: %[[AEXT:.+]] = comb.concat %false, %a : i1, i4
  // CHECK-NEXT: %[[BEXT:.+]] = comb.concat %false, %b : i1, i4
  // CHECK-NEXT: comb.add %[[AEXT]], %[[BEXT]] {comb.nuw = false} : i5
  %false = hw.constant false
  %0 = comb.add %a, %b : i4
  %1 = comb.concat %false, %a : i1, i4
  %2 = comb.concat %false, %b : i1, i4
  %3 = comb.add %1, %2 : i5
  hw.output %0, %3 : i4, i5
}

// CHECK-LABEL: @basic_mul
hw.module @basic_mul(in %a : i4, in %b : i4, out mul7 : i7, out mul8 : i8) {
  // CHECK-NEXT: %c0_i3 = hw.constant 0 : i3
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %[[AEXT7:.+]] = comb.concat %c0_i3, %a : i3, i4
  // CHECK-NEXT: %[[BEXT7:.+]] = comb.concat %c0_i3, %b : i3, i4
  // CHECK-NEXT: comb.mul %[[AEXT7]], %[[BEXT7]] {comb.nuw = true} : i7
  // CHECK-NEXT: %[[AEXT8:.+]] = comb.concat %c0_i4, %a : i4, i4
  // CHECK-NEXT: %[[BEXT8:.+]] = comb.concat %c0_i4, %b : i4, i4
  // CHECK-NEXT: comb.mul %[[AEXT8]], %[[BEXT8]] {comb.nuw = false} : i8
  %c0_i3 = hw.constant 0 : i3
  %c0_i4 = hw.constant 0 : i4
  %1 = comb.concat %c0_i3, %a : i3, i4
  %2 = comb.concat %c0_i3, %b : i3, i4
  %3 = comb.mul %1, %2 : i7
  %4 = comb.concat %c0_i4, %a : i4, i4
  %5 = comb.concat %c0_i4, %b : i4, i4
  %6 = comb.mul %4, %5 : i8
  hw.output %3, %6 : i7, i8
}

// CHECK-LABEL: @basic_fma
hw.module @basic_fma(in %a : i4, in %b : i4, in %c : i4, in %d : i5, out fma1 : i8, out fma2 : i8) {
  // CHECK-NEXT: %c0_i3 = hw.constant 0 : i3
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %[[AEXT:.+]] = comb.concat %c0_i4, %a : i4, i4
  // CHECK-NEXT: %[[BEXT:.+]] = comb.concat %c0_i4, %b : i4, i4
  // CHECK-NEXT: %[[CEXT:.+]] = comb.concat %c0_i4, %c : i4, i4
  // CHECK-NEXT: %[[DEXT:.+]] = comb.concat %c0_i3, %d : i3, i5
  // CHECK-NEXT: %[[MUL:.+]] = comb.mul %[[AEXT]], %[[BEXT]] {comb.nuw = false} : i8
  // Should not overflow when adding 4-bit c
  // CHECK-NEXT: comb.add %[[MUL]], %[[CEXT]] {comb.nuw = false} : i8
  // Should overflow when adding 5-bit d
  // CHECK-NEXT: comb.add %[[MUL]], %[[DEXT]] {comb.nuw = true} : i8
  %c0_i3 = hw.constant 0 : i3
  %c0_i4 = hw.constant 0 : i4
  %4 = comb.concat %c0_i4, %a : i4, i4
  %5 = comb.concat %c0_i4, %b : i4, i4
  %6 = comb.concat %c0_i4, %c : i4, i4
  %7 = comb.concat %c0_i3, %d : i3, i5
  %8 = comb.mul %4, %5 : i8
  %9 = comb.add %8, %6 : i8
  %10 = comb.add %8, %7 : i8
  hw.output %9, %10 : i8, i8
}
