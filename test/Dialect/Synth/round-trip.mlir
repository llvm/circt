// RUN: circt-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s
// CHECK-LABEL: @And
// CHECK-NEXT: %[[RES0:.+]] = synth.aig.and_inv %b, %b : i4
// CHECK-NEXT: %[[RES1:.+]] = synth.aig.and_inv %b, not %b : i4
// CHECK-NEXT: %[[RES2:.+]] = synth.aig.and_inv not %a, not %a : i1
hw.module @And(in %a: i1, in %b: i4) {
  %0 = synth.aig.and_inv %b, %b : i4
  %1 = synth.aig.and_inv %b, not %b : i4
  %2 = synth.aig.and_inv not %a, not %a : i1
}

// CHECK-LABEL: @choice
// CHECK-NEXT: %[[R0:.+]] = synth.choice %a : i4
// CHECK-NEXT: %[[R1:.+]] = synth.choice %a, %b, %a : i4
hw.module @choice(in %a: i4, in %b: i4) {
  %0 = synth.choice %a : i4
  %1 = synth.choice %a, %b, %a : i4
}
