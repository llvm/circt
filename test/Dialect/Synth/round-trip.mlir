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

// CHECK-LABEL: @xor_inv
// CHECK-NEXT: %[[R0:.+]] = synth.xor_inv %a, not %b, %c : i4
// CHECK-NEXT: %[[R1:.+]] = synth.xor_inv not %d : i1
hw.module @xor_inv(in %a: i4, in %b: i4, in %c: i4, in %d: i1) {
  %0 = synth.xor_inv %a, not %b, %c : i4
  %1 = synth.xor_inv not %d : i1
}

// CHECK-LABEL: @dot
// CHECK-NEXT: %[[R0:.+]] = synth.dot %x, not %y, %z : i1
hw.module @dot(in %x: i1, in %y: i1, in %z: i1) {
  %0 = synth.dot %x, not %y, %z : i1
}
