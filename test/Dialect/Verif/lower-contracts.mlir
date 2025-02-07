// RUN: circt-opt --lower-contracts --simplify-assume-eq --canonicalize --cse %s | FileCheck %s

// CHECK: hw.module @Mul9(in %a : i42, out z : i42) {
// CHECK-NEXT:   %c9_i42 = hw.constant 9 : i42
// CHECK-NEXT:   %0 = comb.mul %a, %c9_i42 : i42
// CHECK-NEXT:   hw.output %0 : i42
// CHECK-NEXT: }

// CHECK: verif.formal @Mul9_CheckContract_0 {
// CHECK-NEXT:   %c0_i3 = hw.constant 0 : i3
// CHECK-NEXT:   %c9_i42 = hw.constant 9 : i42
// CHECK-NEXT:   %0 = verif.symbolic_value : i42
// CHECK-NEXT:   %1 = comb.extract %0 from 0 : (i42) -> i39
// CHECK-NEXT:   %2 = comb.concat %1, %c0_i3 : i39, i3
// CHECK-NEXT:   %3 = comb.add %0, %2 : i42
// CHECK-NEXT:   %4 = comb.mul %0, %c9_i42 : i42
// CHECK-NEXT:   %5 = comb.icmp eq %3, %4 : i42
// CHECK-NEXT:   verif.assert %5 : i1
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

// CHECK: hw.module @CarrySaveCompress3to2(in %a0 : i42, in %a1 : i42, in %a2 : i42, out z0 : i42, out z1 : i42) {
// CHECK-NEXT:   %0 = verif.symbolic_value : i42
// CHECK-NEXT:   %1 = verif.symbolic_value : i42
// CHECK-NEXT:   %2 = comb.add %a0, %a1, %a2 : i42
// CHECK-NEXT:   %3 = comb.add %0, %1 : i42
// CHECK-NEXT:   %4 = comb.icmp eq %2, %3 : i42
// CHECK-NEXT:   verif.assume %4 : i1
// CHECK-NEXT:   hw.output %0, %1 : i42, i42
// CHECK-NEXT: }

// CHECK: verif.formal @CarrySaveCompress3to2_CheckContract_0 {
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    %0 = verif.symbolic_value : i42
// CHECK-NEXT:    %1 = verif.symbolic_value : i42
// CHECK-NEXT:    %2 = verif.symbolic_value : i42
// CHECK-NEXT:    %3 = comb.extract %0 from 0 : (i42) -> i41
// CHECK-NEXT:    %4 = comb.extract %1 from 0 : (i42) -> i41
// CHECK-NEXT:    %5 = comb.or %3, %4 : i41
// CHECK-NEXT:    %6 = comb.extract %2 from 0 : (i42) -> i41
// CHECK-NEXT:    %7 = comb.and %5, %6 : i41
// CHECK-NEXT:    %8 = comb.and %3, %4 : i41
// CHECK-NEXT:    %9 = comb.or %8, %7 : i41
// CHECK-NEXT:    %10 = comb.concat %9, %false : i41, i1
// CHECK-NEXT:    %11 = comb.xor %0, %1, %2 : i42
// CHECK-NEXT:    %12 = comb.add %0, %1, %2 : i42
// CHECK-NEXT:    %13 = comb.add %11, %10 : i42
// CHECK-NEXT:    %14 = comb.icmp eq %12, %13 : i42
// CHECK-NEXT:    verif.assert %14 : i1
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

// CHECK: hw.module @ShiftLeft(in %a : i8, in %b : i8, out z : i8) {
// CHECK-NEXT:   %c8_i8 = hw.constant 8 : i8
// CHECK-NEXT:   %0 = comb.icmp ult %b, %c8_i8 : i8
// CHECK-NEXT:   verif.assert %0 : i1
// CHECK-NEXT:   %1 = comb.shl %a, %b : i8
// CHECK-NEXT:   hw.output %1 : i8
// CHECK-NEXT: }

// CHECK:  verif.formal @ShiftLeft_CheckContract_0 {} {
// CHECK-NEXT:   %false = hw.constant false
// CHECK-NEXT:   %c0_i2 = hw.constant 0 : i2
// CHECK-NEXT:   %c0_i4 = hw.constant 0 : i4
// CHECK-NEXT:   %c8_i8 = hw.constant 8 : i8
// CHECK-NEXT:   %0 = verif.symbolic_value : i8
// CHECK-NEXT:   %1 = verif.symbolic_value : i8
// CHECK-NEXT:   %2 = comb.extract %1 from 0 : (i8) -> i4
// CHECK-NEXT:   %3 = comb.concat %2, %c0_i4 : i4, i4
// CHECK-NEXT:   %4 = comb.extract %0 from 2 : (i8) -> i1
// CHECK-NEXT:   %5 = comb.mux %4, %3, %1 : i8
// CHECK-NEXT:   %6 = comb.extract %5 from 0 : (i8) -> i6
// CHECK-NEXT:   %7 = comb.concat %6, %c0_i2 : i6, i2
// CHECK-NEXT:   %8 = comb.extract %0 from 1 : (i8) -> i1
// CHECK-NEXT:   %9 = comb.mux %8, %7, %5 : i8
// CHECK-NEXT:   %10 = comb.extract %9 from 0 : (i8) -> i7
// CHECK-NEXT:   %11 = comb.concat %10, %false : i7, i1
// CHECK-NEXT:   %12 = comb.extract %0 from 0 : (i8) -> i1
// CHECK-NEXT:   %13 = comb.mux %12, %11, %9 : i8
// CHECK-NEXT:   %14 = comb.icmp ult %0, %c8_i8 : i8
// CHECK-NEXT:   verif.assume %14 : i1
// CHECK-NEXT:   %15 = comb.shl %1, %0 : i8
// CHECK-NEXT:   %16 = comb.icmp eq %13, %15 : i8
// CHECK-NEXT:   verif.assert %16 : i1
// CHECK-NEXT: }

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
// CHECK-NEXT:   %c0_i3 = hw.constant 0 : i3
// CHECK-NEXT:   %0 = comb.extract %a from 0 : (i42) -> i39
// CHECK-NEXT:   %1 = comb.concat %0, %c0_i3 : i39, i3
// CHECK-NEXT:   %2 = comb.add %a, %1 : i42
// CHECK-NEXT:   hw.output %2 : i42
// CHECK-NEXT: }

hw.module @NoContract(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42
  hw.output %1 : i42
}

// CHECK: hw.module @TwoContracts(in %a : i42, out z : i42) {
// CHECK-NEXT:   %false = hw.constant false
// CHECK-NEXT:   %c2_i42 = hw.constant 2 : i42
// CHECK-NEXT:   %0 = comb.icmp ult %a, %c2_i42 : i42
// CHECK-NEXT:   verif.assert %0 : i1
// CHECK-NEXT:   %1 = comb.extract %a from 0 : (i42) -> i41
// CHECK-NEXT:   %2 = comb.concat %1, %false : i41, i1
// CHECK-NEXT:   hw.output %2 : i42
// CHECK-NEXT: }

// CHECK: verif.formal @TwoContracts_CheckContract_0 {} {
// CHECK-NEXT:   %c2_i42 = hw.constant 2 : i42
// CHECK-NEXT:   %0 = verif.symbolic_value : i42
// CHECK-NEXT:   %1 = comb.extract %0 from 0 : (i42) -> i41
// CHECK-NEXT:   %2 = comb.icmp ult %0, %c2_i42 : i42
// CHECK-NEXT:   verif.assume %2 : i1
// CHECK-NEXT:   %3 = comb.icmp eq %1, %1 : i41
// CHECK-NEXT:   verif.assert %3 : i1
// CHECK-NEXT: }

// CHECK: verif.formal @TwoContracts_CheckContract_1 {} {
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    %c2_i42 = hw.constant 2 : i42
// CHECK-NEXT:    %0 = verif.symbolic_value : i42
// CHECK-NEXT:    %1 = verif.symbolic_value : i42
// CHECK-NEXT:    %2 = comb.icmp ult %0, %c2_i42 : i42
// CHECK-NEXT:    verif.assert %2 : i1
// CHECK-NEXT:    %3 = comb.extract %0 from 0 : (i42) -> i41
// CHECK-NEXT:    %4 = comb.concat %3, %false : i41, i1
// CHECK-NEXT:    %5 = comb.icmp eq %1, %4 : i42
// CHECK-NEXT:    verif.assume %5 : i1
// CHECK-NEXT:    verif.assert %5 : i1
// CHECK-NEXT: }

hw.module @TwoContracts(in %a: i42, out z: i42) {
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.shl %a, %c1_i42 : i42
  %1 = verif.contract %0 : i42 {
    %c2_i42 = hw.constant 2 : i42
    %req = comb.icmp ult %a, %c2_i42 : i42
    verif.require %req : i1
    %2 = comb.mul %a, %c2_i42 : i42
    %3 = comb.icmp eq %1, %2 : i42
    verif.ensure %3 : i1
  }
  %4 = verif.contract %1 : i42 {
    %5 = comb.add %a, %a : i42
    %6 = comb.icmp eq %4, %5 : i42
    verif.ensure %6 : i1
  }
  hw.output %4 : i42
}

// CHECK: hw.module @NestedContract(in %a : i42, in %b : i42, in %s : i1, out z : i42) {
// CHECK-NEXT:   %false = hw.constant false
// CHECK-NEXT:   %0 = comb.extract %a from 0 : (i42) -> i41
// CHECK-NEXT:   %1 = comb.concat %0, %false : i41, i1
// CHECK-NEXT:   %2 = comb.mul %a, %a : i42
// CHECK-NEXT:   %3 = scf.if %s -> (i42) {
// CHECK-NEXT:     %6 = comb.add %1, %2 : i42
// CHECK-NEXT:     scf.yield %6 : i42
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %6 = comb.mul %1, %2 : i42
// CHECK-NEXT:     scf.yield %6 : i42
// CHECK-NEXT:   }
// CHECK-NEXT:   %4 = verif.symbolic_value : i42
// CHECK-NEXT:   %5 = comb.icmp eq %3, %b : i42
// CHECK-NEXT:   verif.assume %5 : i1
// CHECK-NEXT:   hw.output %3 : i42
// CHECK-NEXT: }

// CHECK: verif.formal @NestedContract_CheckContract_0 {} {
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    %0 = verif.symbolic_value : i42
// CHECK-NEXT:    %1 = verif.symbolic_value : i1
// CHECK-NEXT:    %2 = verif.symbolic_value : i42
// CHECK-NEXT:    %3 = comb.mul %2, %2 : i42
// CHECK-NEXT:    %4 = comb.extract %2 from 0 : (i42) -> i41
// CHECK-NEXT:    %5 = comb.concat %4, %false : i41, i1
// CHECK-NEXT:    %6 = scf.if %1 -> (i42) {
// CHECK-NEXT:      %8 = comb.add %5, %3 : i42
// CHECK-NEXT:      scf.yield %8 : i42
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %8 = comb.mul %5, %3 : i42
// CHECK-NEXT:      scf.yield %8 : i42
// CHECK-NEXT:    }
// CHECK-NEXT:    %7 = comb.icmp eq %6, %0 : i42
// CHECK-NEXT:    verif.assert %7 : i1
// CHECK-NEXT: }

hw.module @NestedContract(in %a: i42, in %b: i42, in %s: i1, out z: i42) {
  %0 = comb.add %a, %a : i42
  %1 = comb.mul %a, %a : i42
  %2 = scf.if %s -> (i42) {
    %3 = comb.add %0, %1 : i42
    scf.yield %3 : i42
  } else {
    %3 = comb.mul %0, %1 : i42
    scf.yield %3 : i42
  }
  %3 = verif.contract %2 : i42 {
    %4 = comb.icmp eq %2, %b : i42
    verif.ensure %4 : i1
  }
  hw.output %2 : i42
}
