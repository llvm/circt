// RUN: circt-opt %s --comb-int-range-narrowing --canonicalize | FileCheck %s

// CHECK-LABEL: @basic_csa
hw.module @basic_csa(in %a : i1, in %b : i1, in %c : i1, out add_abc : i3) {
// CHECK-NEXT: %false = hw.constant false
// CHECK-NEXT: %0 = comb.concat %false, %a : i1, i1
// CHECK-NEXT: %1 = comb.concat %false, %b : i1, i1
// CHECK-NEXT: %2 = comb.concat %false, %c : i1, i1
// CHECK-NEXT: %3 = comb.add %0, %1, %2 : i2
// CHECK-NEXT: %4 = comb.concat %false, %3 : i1, i2
// CHECK-NEXT: hw.output %4 : i3
  %c0_i2 = hw.constant 0 : i2
  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i1
  %1 = comb.concat %false, %b : i1, i1
  %2 = comb.add %0, %1 : i2
  %3 = comb.concat %false, %2 : i1, i2
  %4 = comb.concat %c0_i2, %c : i2, i1
  %5 = comb.add %3, %4 : i3
  hw.output %5 : i3
}
