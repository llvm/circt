// RUN: circt-opt %s --synth-lower-variadic | FileCheck %s
// CHECK: hw.module @Basic
hw.module @Basic(in %a: i2, in %b: i2, in %c: i2, in %d: i2, in %e: i2, out f: i2) {
  // CHECK:      %[[RES0:.+]] = synth.aig.and_inv not %a, %b : i2
  // CHECK-NEXT: %[[RES1:.+]] = synth.aig.and_inv not %d, %e : i2
  // CHECK-NEXT: %[[RES2:.+]] = synth.aig.and_inv %c, %[[RES1]] : i2
  // CHECK-NEXT: %[[RES3:.+]] = synth.aig.and_inv %[[RES0]], %[[RES2]] : i2
  // CHECK-NEXT: hw.output %[[RES3]] : i2
  %0 = synth.aig.and_inv not %a, %b, %c, not %d, %e : i2
  hw.output %0 : i2
}
