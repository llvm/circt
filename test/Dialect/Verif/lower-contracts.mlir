// RUN: circt-opt --lower-contracts %s | FileCheck %s

// CHECK: hw.module @Mul9(in %a : i42, out z : i42) {
// CHECK-NEXT:  %c3_i42 = hw.constant 3 : i42
// CHECK-NEXT:  %c9_i42 = hw.constant 9 : i42
// CHECK-NEXT:  %0 = comb.shl %a, %c3_i42 : i42
// CHECK-NEXT:  %1 = comb.add %a, %0 : i42
// CHECK-NEXT:  %2 = verif.contract %1 : i42 {
// CHECK-NEXT:    %3 = comb.mul %a, %c9_i42 : i42
// CHECK-NEXT:    %4 = comb.icmp eq %2, %3 : i42
// CHECK-NEXT:    verif.ensure %4 : i1
// CHECK-NEXT:  }
// CHECK-NEXT:  hw.output %2 : i42
// CHECK-NEXT: }

// CHECK: verif.formal @Mul9_CheckContract_0 {
// CHECK-NEXT:    %0 = verif.symbolic_value : i42
// CHECK-NEXT:    %c3_i42 = hw.constant 3 : i42
// CHECK-NEXT:    %1 = comb.shl %0, %c3_i42 : i42
// CHECK-NEXT:    %2 = comb.add %0, %1 : i42
// CHECK-NEXT:    %c9_i42 = hw.constant 9 : i42
// CHECK-NEXT:    %3 = comb.mul %0, %c9_i42 : i42
// CHECK-NEXT:    %4 = comb.icmp eq %2, %3 : i42
// CHECK-NEXT:    verif.assert %4 : i1
// CHECK-NEXT: }

hw.module @Mul9(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %c9_i42 = hw.constant 9 : i42
  %0 = comb.shl %a, %c3_i42 : i42    // 8*a
  %1 = comb.add %a, %0 : i42         // a + 8*a
  %2 = verif.contract %1 : i42 {
    %3 = comb.mul %a, %c9_i42 : i42  // 9*a
    %4 = comb.icmp eq %2, %3 : i42   // 9*a == a + 8*a
    verif.ensure %4 : i1
  }
  hw.output %2 : i42
}

// CHECK: verif.formal @CarrySaveCompress3to2_CheckContract_0 {
// CHECK-NEXT:    %0 = verif.symbolic_value : i42
// CHECK-NEXT:    %1 = verif.symbolic_value : i42
// CHECK-NEXT:    %2 = verif.symbolic_value : i42
// CHECK-NEXT:    %3 = comb.xor %0, %1, %2 : i42
// CHECK-NEXT:    %4 = comb.and %0, %1 : i42
// CHECK-NEXT:    %5 = comb.or %0, %1 : i42
// CHECK-NEXT:    %6 = comb.and %5, %2 : i42
// CHECK-NEXT:    %7 = comb.or %4, %6 : i42
// CHECK-NEXT:    %c1_i42 = hw.constant 1 : i42
// CHECK-NEXT:    %8 = comb.shl %7, %c1_i42 : i42
// CHECK-NEXT:    %9 = comb.add %0, %1, %2 : i42
// CHECK-NEXT:    %10 = comb.add %3, %8 : i42
// CHECK-NEXT:    %11 = comb.icmp eq %9, %10 : i42
// CHECK-NEXT:    verif.assert %11 : i1
// CHECK-NEXT: }

hw.module @CarrySaveCompress3to2(
  in %a0: i42, in %a1: i42, in %a2: i42,
  out z0: i42, out z1: i42
) {
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.xor %a0, %a1, %a2 : i42  // sum bits of FA (a0^a1^a2)
  %1 = comb.and %a0, %a1 : i42
  %2 = comb.or %a0, %a1 : i42
  %3 = comb.and %2, %a2 : i42
  %4 = comb.or %1, %3 : i42          // carry bits of FA (a0&a1 | a2&(a0|a1))
  %5 = comb.shl %4, %c1_i42 : i42    // %5 = carry << 1
  %z0, %z1 = verif.contract %0, %5 : i42, i42 {
    %inputSum = comb.add %a0, %a1, %a2 : i42
    %outputSum = comb.add %z0, %z1 : i42
    %6 = comb.icmp eq %inputSum, %outputSum : i42
    verif.ensure %6 : i1
  }
  hw.output %z0, %z1 : i42, i42
}

// CHECK:  verif.formal @ShiftLeft_CheckContract_0 {} {
// CHECK-NEXT:    %0 = verif.symbolic_value : i8
// CHECK-NEXT:    %1 = comb.extract %0 from 0 : (i8) -> i1
// CHECK-NEXT:    %2 = comb.extract %0 from 1 : (i8) -> i1
// CHECK-NEXT:    %3 = comb.extract %0 from 2 : (i8) -> i1
// CHECK-NEXT:    %4 = verif.symbolic_value : i8
// CHECK-NEXT:    %c4_i8 = hw.constant 4 : i8
// CHECK-NEXT:    %5 = comb.shl %4, %c4_i8 : i8
// CHECK-NEXT:    %6 = comb.mux %3, %5, %4 : i8
// CHECK-NEXT:    %c2_i8 = hw.constant 2 : i8
// CHECK-NEXT:    %7 = comb.shl %6, %c2_i8 : i8
// CHECK-NEXT:    %8 = comb.mux %2, %7, %6 : i8
// CHECK-NEXT:    %c1_i8 = hw.constant 1 : i8
// CHECK-NEXT:    %9 = comb.shl %8, %c1_i8 : i8
// CHECK-NEXT:    %10 = comb.mux %1, %9, %8 : i8
// CHECK-NEXT:    %c8_i8 = hw.constant 8 : i8
// CHECK-NEXT:    %11 = comb.icmp ult %0, %c8_i8 : i8
// CHECK-NEXT:    verif.assume %11 : i1
// CHECK-NEXT:    %12 = comb.shl %4, %0 : i8
// CHECK-NEXT:    %13 = comb.icmp eq %10, %12 : i8
// CHECK-NEXT:    verif.assert %13 : i1
// CHECK-NEXT:  }

hw.module @ShiftLeft(in %a: i8, in %b: i8, out z: i8) {
  %c4_i8 = hw.constant 4 : i8
  %c2_i8 = hw.constant 2 : i8
  %c1_i8 = hw.constant 1 : i8
  %b2 = comb.extract %b from 2 : (i8) -> i1
  %b1 = comb.extract %b from 1 : (i8) -> i1
  %b0 = comb.extract %b from 0 : (i8) -> i1
  %0 = comb.shl %a, %c4_i8 : i8
  %1 = comb.mux %b2, %0, %a : i8
  %2 = comb.shl %1, %c2_i8 : i8
  %3 = comb.mux %b1, %2, %1 : i8
  %4 = comb.shl %3, %c1_i8 : i8
  %5 = comb.mux %b0, %4, %3 : i8

  // Contract to check that the multiplexers and constant shifts above indeed
  // produce the correct shift by 0 to 7 places, assuming the shift amount is
  // less than 8 (we can't shift a number out).
  %z = verif.contract %5 : i8 {
    // Shift amount must be less than 8.
    %c8_i8 = hw.constant 8 : i8
    %blt8 = comb.icmp ult %b, %c8_i8 : i8
    verif.require %blt8 : i1

    // In that case the mux tree computes the correct left-shift.
    %ashl = comb.shl %a, %b : i8
    %eq = comb.icmp eq %z, %ashl : i8
    verif.ensure %eq : i1
  }

  hw.output %z : i8
}

// CHECK: hw.module @NoContract(in %a : i42, out z : i42) {
// CHECK-NEXT:   %c3_i42 = hw.constant 3 : i42
// CHECK-NEXT:   %0 = comb.shl %a, %c3_i42 : i42
// CHECK-NEXT:   %1 = comb.add %a, %0 : i42
// CHECK-NEXT:   hw.output %1 : i42
// CHECK-NEXT: }

hw.module @NoContract(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42
  hw.output %1 : i42
}
