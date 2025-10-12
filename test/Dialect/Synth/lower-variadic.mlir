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
