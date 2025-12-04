// RUN: circt-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s
// CHECK-LABEL: @basic
// CHECK: synth.mig.maj_inv
hw.module @basic(in %a : i4, out result : i4) {
    %0 = synth.mig.maj_inv not %a, %a, %a {sv.namehint = "out0"} : i4
    hw.output %0 : i4
}

// CHECK-LABEL: @And
// CHECK-NEXT: %[[RES0:.+]] = synth.aig.and_inv %b, %b : i4
// CHECK-NEXT: %[[RES1:.+]] = synth.aig.and_inv %b, not %b : i4
// CHECK-NEXT: %[[RES2:.+]] = synth.aig.and_inv not %a, not %a : i1
hw.module @And(in %a: i1, in %b: i4) {
  %0 = synth.aig.and_inv %b, %b : i4
  %1 = synth.aig.and_inv %b, not %b : i4
  %2 = synth.aig.and_inv not %a, not %a : i1
}

