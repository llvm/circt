// RUN: circt-opt %s --synth-lower-variadic | FileCheck %s --check-prefixes=COMMON,TIMING
// RUN: circt-opt %s --synth-lower-variadic=timing-aware=false | FileCheck %s --check-prefixes=COMMON,NO-TIMING
// COMMON-LABEL: hw.module @Basic
hw.module @Basic(in %a: i2, in %b: i2, in %c: i2, in %d: i2, in %e: i2, out f: i2) {
  // COMMON-NEXT: %[[RES0:.+]] = synth.aig.and_inv not %a, %b : i2
  // COMMON-NEXT: %[[RES1:.+]] = synth.aig.and_inv %c, not %d : i2
  // COMMON-NEXT: %[[RES2:.+]] = synth.aig.and_inv %e, %[[RES0]] : i2
  // COMMON-NEXT: %[[RES3:.+]] = synth.aig.and_inv %[[RES1]], %[[RES2]] : i2
  %0 = synth.aig.and_inv not %a, %b, %c, not %d, %e : i2
  hw.output %0 : i2
}

// COMMON-LABEL: hw.module @AddMul
hw.module @AddMul(in %x: i4, in %y: i4, in %z: i4, out out: i4) {
  // constant * (x + y) * z
  // => (z * constant) * (x + y)
  // COMMON-NEXT: %c5_i4 = hw.constant 5 : i4
  // COMMON-NEXT: %[[ADD:.+]] = comb.add %x, %y : i4
  // TIMING-NEXT: %[[MUL:.+]] = comb.mul %c5_i4, %z : i4
  // TIMING-NEXT: %[[RES:.+]] = comb.mul %[[ADD]], %[[MUL]] : i4
  // TIMING-NEXT: hw.output %[[RES]] : i4
  // NO-TIMING-NEXT: %[[MUL:.+]] = comb.mul %c5_i4, %[[ADD]] : i4
  // NO-TIMING-NEXT: %[[RES:.+]] = comb.mul %z, %[[MUL]] : i4
  // NO-TIMING-NEXT: hw.output %[[RES]] : i4
  %0 = comb.mul %c_i5, %add, %z : i4
  %c_i5 = hw.constant 5 : i4
  // Check topological sort as well.
  %add = comb.add %x, %y : i4
  hw.output %0 : i4
}

// COMMON-LABEL: hw.module @Tree1
hw.module @Tree1(in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1, in %g: i1, out o1: i1) {
  // COMMON-NEXT: %[[AND_INV0:.+]] = synth.aig.and_inv %d, not %e : i1
  // TIMING-NEXT: %[[AND_INV1:.+]] = synth.aig.and_inv not %c, %f : i1
  // TIMING-NEXT: %[[AND_INV2:.+]] = synth.aig.and_inv not %[[AND_INV0]], %[[AND_INV1]] : i1
  // TIMING-NEXT: %[[AND_INV3:.+]] = synth.aig.and_inv %a, not %b : i1
  // TIMING-NEXT: %[[AND_INV4:.+]] = synth.aig.and_inv %g, %[[AND_INV3]] : i1
  // TIMING-NEXT: %[[AND_INV5:.+]] = synth.aig.and_inv not %[[AND_INV2]], %[[AND_INV4]] : i1
  // TIMING-NEXT: hw.output %[[AND_INV5]] : i1
  // NO-TIMING-NEXT: %[[AND_INV1:.+]] = synth.aig.and_inv not %c, not %[[AND_INV0]] : i1
  // NO-TIMING-NEXT: %[[AND_INV2:.+]] = synth.aig.and_inv %f, %[[AND_INV1]] : i1
  // NO-TIMING-NEXT: %[[AND_INV3:.+]] = synth.aig.and_inv %a, not %b : i1
  // NO-TIMING-NEXT: %[[AND_INV4:.+]] = synth.aig.and_inv not %[[AND_INV2]], %g : i1
  // NO-TIMING-NEXT: %[[AND_INV5:.+]] = synth.aig.and_inv %[[AND_INV3]], %[[AND_INV4]] : i1
  // NO-TIMING-NEXT: hw.output %[[AND_INV5]] : i1
  %0 = synth.aig.and_inv %d, not %e : i1
  %1 = synth.aig.and_inv not %c, not %0, %f : i1
  %2 = synth.aig.and_inv %a, not %b, not %1, %g : i1

  hw.output %2 : i1
}

// COMMON-LABEL: hw.module @ChildRegion
hw.module @ChildRegion(in %x: i4, in %y: i4, in %z: i4) {
  // COMMON-NEXT: %[[TMP:.+]] = comb.or %x, %y : i4
  // COMMON-NEXT: %[[OR:.+]] = comb.or %z, %[[TMP]] : i4
  %0 = comb.or %x, %y, %z : i4
  sv.initial {
    // COMMON: comb.and %[[OR]], %y, %z : i4
    %1 = comb.and %0, %y, %z : i4
  }
}
