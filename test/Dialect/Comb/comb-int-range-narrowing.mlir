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

// CHECK-LABEL: @basic_fma
hw.module @basic_fma(in %a : i4, in %b : i4, in %c : i4, out d : i9) {
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %0 = comb.concat %c0_i4, %a : i4, i4
  // CHECK-NEXT: %1 = comb.concat %c0_i4, %b : i4, i4
  // CHECK-NEXT: %2 = comb.mul %0, %1 : i8
  // CHECK-NEXT: %3 = comb.concat %c0_i4, %c : i4, i4
  // CHECK-NEXT: %4 = comb.add %2, %3 : i8
  // CHECK-NEXT: %5 = comb.concat %false, %4 : i1, i8
  // CHECK-NEXT: hw.output %5 : i9
  %c0_i5 = hw.constant 0 : i5
  %0 = comb.concat %c0_i5, %a : i5, i4
  %1 = comb.concat %c0_i5, %b : i5, i4
  %2 = comb.mul %0, %1 : i9
  %3 = comb.concat %c0_i5, %c : i5, i4
  %4 = comb.add %2, %3 : i9
  hw.output %4 : i9
}

// CHECK-LABEL: @const_sub
hw.module @const_sub(in %a : i8, out sub_res : i10) {
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %c-256_i9 = hw.constant -256 : i9
  // CHECK-NEXT: %0 = comb.concat %false, %a : i1, i8
  // CHECK-NEXT: %1 = comb.sub %c-256_i9, %0 : i9
  // CHECK-NEXT: %2 = comb.concat %false, %1 : i1, i9
  // CHECK-NEXT: hw.output %2 : i10
  %c256_i10 = hw.constant 256 : i10
  %c0_i2 = hw.constant 0 : i2
  %0 = comb.concat %c0_i2, %a : i2, i8
  %1 = comb.sub %c256_i10, %0 : i10
  hw.output %1 : i10
}

// CHECK-LABEL: @do_nothing
hw.module @do_nothing(in %a : i8, in %b : i9, in %c : i10, in %d : i16, out res : i18) {
  // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %c0_i9 = hw.constant 0 : i9
  // CHECK-NEXT: %c0_i8 = hw.constant 0 : i8
  // CHECK-NEXT: %0 = comb.concat %c0_i9, %a : i9, i8
  // CHECK-NEXT: %1 = comb.concat %c0_i8, %b : i8, i9
  // CHECK-NEXT: %2 = comb.mul %0, %1 : i17
  // CHECK-NEXT: %3 = comb.concat %false, %2 : i1, i17
  // CHECK-NEXT: %4 = comb.concat %c0_i8, %c : i8, i10
  // CHECK-NEXT: %5 = comb.concat %c0_i2, %d : i2, i16
  // CHECK-NEXT: %6 = comb.add %3, %4, %5 : i18
  // CHECK-NEXT: hw.output %6 : i18
  %c0_i2 = hw.constant 0 : i2
  %false = hw.constant false
  %c0_i9 = hw.constant 0 : i9
  %c0_i8 = hw.constant 0 : i8
  %0 = comb.concat %c0_i9, %a : i9, i8
  %1 = comb.concat %c0_i8, %b : i8, i9
  %2 = comb.mul %0, %1 : i17
  %3 = comb.concat %false, %2 : i1, i17
  %4 = comb.concat %c0_i8, %c : i8, i10
  %5 = comb.concat %c0_i2, %d : i2, i16
  %6 = comb.add %3, %4, %5 : i18
  hw.output %6 : i18
}

hw.module @logical_ops(in %a : i8, in %b : i9, in %c : i10, in %d : i16, out res : i18) {
  // CHECK-NEXT %c0_i7 = hw.constant 0 : i7
  // CHECK-NEXT %false = hw.constant false
  // CHECK-NEXT %c0_i9 = hw.constant 0 : i9
  // CHECK-NEXT %c0_i8 = hw.constant 0 : i8
  // CHECK-NEXT %0 = comb.concat %c0_i9, %a : i9, i8
  // CHECK-NEXT %1 = comb.concat %c0_i8, %b : i8, i9
  // CHECK-NEXT %2 = comb.and %0, %1 : i17
  // CHECK-NEXT %3 = comb.concat %c0_i7, %c : i7, i10
  // CHECK-NEXT %4 = comb.or %2, %3 : i17
  // CHECK-NEXT %5 = comb.concat %false, %d : i1, i16
  // CHECK-NEXT %6 = comb.add %4, %5 : i17
  // CHECK-NEXT %7 = comb.concat %false, %6 : i1, i17
  // CHECK-NEXT hw.output %7 : i18
  %c0_i2 = hw.constant 0 : i2
  %false = hw.constant false
  %c0_i9 = hw.constant 0 : i9
  %c0_i8 = hw.constant 0 : i8
  %0 = comb.concat %c0_i9, %a : i9, i8
  %1 = comb.concat %c0_i8, %b : i8, i9
  %2 = comb.and %0, %1 : i17
  %3 = comb.concat %false, %2 : i1, i17
  %4 = comb.concat %c0_i8, %c : i8, i10
  %5 = comb.or %3, %4 : i18
  %6 = comb.concat %c0_i2, %d : i2, i16
  %7 = comb.add %5, %6 : i18
  hw.output %7 : i18
}