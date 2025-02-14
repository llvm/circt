// RUN: circt-opt --lower-contracts --simplify-assume-eq --canonicalize --cse %s | FileCheck %s

// CHECK-LABEL: hw.module @Mul9
// CHECK-NEXT:   %c9_i42 = hw.constant 9 : i42
// CHECK-NEXT:   [[TMP:%.+]] = comb.mul %a, %c9_i42 : i42
// CHECK-NEXT:   hw.output [[TMP]] : i42
// CHECK-NEXT: }

// CHECK-LABEL: verif.formal @Mul9_CheckContract_0
// CHECK-NEXT:   %c0_i3 = hw.constant 0 : i3
// CHECK-NEXT:   %c9_i42 = hw.constant 9 : i42
// CHECK-NEXT:   [[A:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:   [[TMP1:%.+]] = comb.extract [[A]] from 0 : (i42) -> i39
// CHECK-NEXT:   [[TMP2:%.+]] = comb.concat [[TMP1]], %c0_i3 : i39, i3
// CHECK-NEXT:   [[TMP1:%.+]] = comb.add [[A]], [[TMP2]] : i42
// CHECK-NEXT:   [[TMP2:%.+]] = comb.mul [[A]], %c9_i42 : i42
// CHECK-NEXT:   [[TMP3:%.+]] = comb.icmp eq [[TMP1]], [[TMP2]] : i42
// CHECK-NEXT:   verif.assert [[TMP3]] : i1
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

// CHECK-LABEL: hw.module @CarrySaveCompress3to2
// CHECK-NEXT:   [[Z0:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:   [[Z1:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:   [[TMP1:%.+]] = comb.add %a0, %a1, %a2 : i42
// CHECK-NEXT:   [[TMP2:%.+]] = comb.add [[Z0]], [[Z1]] : i42
// CHECK-NEXT:   [[TMP3:%.+]] = comb.icmp eq [[TMP1]], [[TMP2]] : i42
// CHECK-NEXT:   verif.assume [[TMP3]] : i1
// CHECK-NEXT:   hw.output [[Z0]], [[Z1]] : i42, i42
// CHECK-NEXT: }

// CHECK-LABEL: verif.formal @CarrySaveCompress3to2_CheckContract_0
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    [[A0:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:    [[A1:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:    [[A2:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:    [[TMP1:%.+]] = comb.extract [[A0]] from 0 : (i42) -> i41
// CHECK-NEXT:    [[TMP2:%.+]] = comb.extract [[A1]] from 0 : (i42) -> i41
// CHECK-NEXT:    [[TMP3:%.+]] = comb.or [[TMP1]], [[TMP2]] : i41
// CHECK-NEXT:    [[TMP4:%.+]] = comb.extract [[A2]] from 0 : (i42) -> i41
// CHECK-NEXT:    [[TMP5:%.+]] = comb.and [[TMP3]], [[TMP4]] : i41
// CHECK-NEXT:    [[TMP3:%.+]] = comb.and [[TMP1]], [[TMP2]] : i41
// CHECK-NEXT:    [[TMP4:%.+]] = comb.or [[TMP3]], [[TMP5]] : i41
// CHECK-NEXT:    [[TMP1:%.+]] = comb.concat [[TMP4]], %false : i41, i1
// CHECK-NEXT:    [[TMP2:%.+]] = comb.xor [[A0]], [[A1]], [[A2]] : i42
// CHECK-NEXT:    [[TMP3:%.+]] = comb.add [[A0]], [[A1]], [[A2]] : i42
// CHECK-NEXT:    [[TMP4:%.+]] = comb.add [[TMP2]], [[TMP1]] : i42
// CHECK-NEXT:    [[TMP1:%.+]] = comb.icmp eq [[TMP3]], [[TMP4]] : i42
// CHECK-NEXT:    verif.assert [[TMP1]] : i1
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

// CHECK-LABEL: hw.module @ShiftLeft
// CHECK-NEXT:   %c8_i8 = hw.constant 8 : i8
// CHECK-NEXT:   [[TMP0:%.+]] = comb.icmp ult %b, %c8_i8 : i8
// CHECK-NEXT:   verif.assert [[TMP0]] : i1
// CHECK-NEXT:   [[TMP1:%.+]] = comb.shl %a, %b : i8
// CHECK-NEXT:   hw.output [[TMP1]] : i8
// CHECK-NEXT: }

// CHECK-LABEL:  verif.formal @ShiftLeft_CheckContract_0
// CHECK-NEXT:   %false = hw.constant false
// CHECK-NEXT:   %c0_i2 = hw.constant 0 : i2
// CHECK-NEXT:   %c0_i4 = hw.constant 0 : i4
// CHECK-NEXT:   %c8_i8 = hw.constant 8 : i8
// CHECK-NEXT:   [[TMP0:%.+]] = verif.symbolic_value : i8
// CHECK-NEXT:   [[TMP1:%.+]] = verif.symbolic_value : i8
// CHECK-NEXT:   [[TMP2:%.+]] = comb.extract [[TMP1]] from 0 : (i8) -> i4
// CHECK-NEXT:   [[TMP3:%.+]] = comb.concat [[TMP2]], %c0_i4 : i4, i4
// CHECK-NEXT:   [[TMP4:%.+]] = comb.extract [[TMP0]] from 2 : (i8) -> i1
// CHECK-NEXT:   [[TMP5:%.+]] = comb.mux [[TMP4]], [[TMP3]], [[TMP1]] : i8
// CHECK-NEXT:   [[TMP6:%.+]] = comb.extract [[TMP5]] from 0 : (i8) -> i6
// CHECK-NEXT:   [[TMP7:%.+]] = comb.concat [[TMP6]], %c0_i2 : i6, i2
// CHECK-NEXT:   [[TMP8:%.+]] = comb.extract [[TMP0]] from 1 : (i8) -> i1
// CHECK-NEXT:   [[TMP9:%.+]] = comb.mux [[TMP8]], [[TMP7]], [[TMP5]] : i8
// CHECK-NEXT:   [[TMP10:%.+]] = comb.extract [[TMP9]] from 0 : (i8) -> i7
// CHECK-NEXT:   [[TMP11:%.+]] = comb.concat [[TMP10]], %false : i7, i1
// CHECK-NEXT:   [[TMP12:%.+]] = comb.extract [[TMP0]] from 0 : (i8) -> i1
// CHECK-NEXT:   [[TMP13:%.+]] = comb.mux [[TMP12]], [[TMP11]], [[TMP9]] : i8
// CHECK-NEXT:   [[TMP14:%.+]] = comb.icmp ult [[TMP0]], %c8_i8 : i8
// CHECK-NEXT:   verif.assume [[TMP14]] : i1
// CHECK-NEXT:   [[TMP15:%.+]] = comb.shl [[TMP1]], [[TMP0]] : i8
// CHECK-NEXT:   [[TMP16:%.+]] = comb.icmp eq [[TMP13]], [[TMP15]] : i8
// CHECK-NEXT:   verif.assert [[TMP16]] : i1
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

// CHECK-LABEL: hw.module @NoContract
// CHECK-NOT: verif.formal

hw.module @NoContract(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42
  hw.output %1 : i42
}

// CHECK-LABEL: hw.module @TwoContracts
// CHECK-NEXT:   %false = hw.constant false
// CHECK-NEXT:   %c2_i42 = hw.constant 2 : i42
// CHECK-NEXT:   [[TMP0:%.+]] = comb.icmp ult %a, %c2_i42 : i42
// CHECK-NEXT:   verif.assert [[TMP0]] : i1
// CHECK-NEXT:   [[TMP1:%.+]] = comb.extract %a from 0 : (i42) -> i41
// CHECK-NEXT:   [[TMP2:%.+]] = comb.concat [[TMP1]], %false : i41, i1
// CHECK-NEXT:   hw.output [[TMP2]] : i42
// CHECK-NEXT: }

// CHECK-LABEL: verif.formal @TwoContracts_CheckContract_0
// CHECK-NEXT:   %c2_i42 = hw.constant 2 : i42
// CHECK-NEXT:   [[TMP0:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:   [[TMP1:%.+]] = comb.extract [[TMP0]] from 0 : (i42) -> i41
// CHECK-NEXT:   [[TMP2:%.+]] = comb.icmp ult [[TMP0]], %c2_i42 : i42
// CHECK-NEXT:   verif.assume [[TMP2]] : i1
// CHECK-NEXT:   [[TMP3:%.+]] = comb.icmp eq [[TMP1]], [[TMP1]] : i41
// CHECK-NEXT:   verif.assert [[TMP3]] : i1
// CHECK-NEXT: }

// CHECK-LABEL: verif.formal @TwoContracts_CheckContract_1
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    %c2_i42 = hw.constant 2 : i42
// CHECK-NEXT:    [[TMP0:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:    [[TMP1:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:    [[TMP2:%.+]] = comb.icmp ult [[TMP0]], %c2_i42 : i42
// CHECK-NEXT:    verif.assert [[TMP2]] : i1
// CHECK-NEXT:    [[TMP3:%.+]] = comb.extract [[TMP0]] from 0 : (i42) -> i41
// CHECK-NEXT:    [[TMP4:%.+]] = comb.concat [[TMP3]], %false : i41, i1
// CHECK-NEXT:    [[TMP5:%.+]] = comb.icmp eq [[TMP1]], [[TMP4]] : i42
// CHECK-NEXT:    verif.assume [[TMP5]] : i1
// CHECK-NEXT:    verif.assert [[TMP5]] : i1
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

// CHECK-LABEL: hw.module @NestedContract
// CHECK-NEXT:   %false = hw.constant false
// CHECK-NEXT:   [[TMP0:%.+]] = comb.extract %a from 0 : (i42) -> i41
// CHECK-NEXT:   [[TMP1:%.+]] = comb.concat [[TMP0]], %false : i41, i1
// CHECK-NEXT:   [[TMP2:%.+]] = comb.mul %a, %a : i42
// CHECK-NEXT:   [[TMP3:%.+]] = scf.if %s -> (i42) {
// CHECK-NEXT:     [[TMP6:%.+]] = comb.add [[TMP1]], [[TMP2]] : i42
// CHECK-NEXT:     scf.yield [[TMP6]] : i42
// CHECK-NEXT:   } else {
// CHECK-NEXT:     [[TMP6:%.+]] = comb.mul [[TMP1]], [[TMP2]] : i42
// CHECK-NEXT:     scf.yield [[TMP6]] : i42
// CHECK-NEXT:   }
// CHECK-NEXT:   [[TMP5:%.+]] = comb.icmp eq [[TMP3]], %b : i42
// CHECK-NEXT:   verif.assume [[TMP5]] : i1
// CHECK-NEXT:   hw.output [[TMP3]] : i42
// CHECK-NEXT: }

// CHECK-LABEL: verif.formal @NestedContract_CheckContract_0
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    [[TMP0:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:    [[TMP1:%.+]] = verif.symbolic_value : i1
// CHECK-NEXT:    [[TMP2:%.+]] = verif.symbolic_value : i42
// CHECK-NEXT:    [[TMP3:%.+]] = comb.mul [[TMP2]], [[TMP2]] : i42
// CHECK-NEXT:    [[TMP4:%.+]] = comb.extract [[TMP2]] from 0 : (i42) -> i41
// CHECK-NEXT:    [[TMP5:%.+]] = comb.concat [[TMP4]], %false : i41, i1
// CHECK-NEXT:    [[TMP6:%.+]] = scf.if [[TMP1]] -> (i42) {
// CHECK-NEXT:      [[TMP8:%.+]] = comb.add [[TMP5]], [[TMP3]] : i42
// CHECK-NEXT:      scf.yield [[TMP8]] : i42
// CHECK-NEXT:    } else {
// CHECK-NEXT:      [[TMP8:%.+]] = comb.mul [[TMP5]], [[TMP3]] : i42
// CHECK-NEXT:      scf.yield [[TMP8]] : i42
// CHECK-NEXT:    }
// CHECK-NEXT:    [[TMP7:%.+]] = comb.icmp eq [[TMP6]], [[TMP0]] : i42
// CHECK-NEXT:    verif.assert [[TMP7]] : i1
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
