// RUN: circt-opt %s --externalize-clock-gate --verify-diagnostics | FileCheck %s --check-prefixes=CHECK,CHECK-DEFAULT
// RUN: circt-opt %s --externalize-clock-gate="name=SuchClock input=CI output=CO enable=EN test-enable=TEN instance-name=gated" --verify-diagnostics | FileCheck %s --check-prefixes=CHECK,CHECK-CUSTOM
// RUN: circt-opt %s --externalize-clock-gate="name=VeryGate test-enable=" --verify-diagnostics | FileCheck %s --check-prefixes=CHECK,CHECK-WITHOUT-TESTENABLE

// CHECK-DEFAULT: hw.module.extern @CKG(%I: i1, %E: i1, %TE: i1) -> (O: i1)
// CHECK-CUSTOM: hw.module.extern @SuchClock(%CI: i1, %EN: i1, %TEN: i1) -> (CO: i1)
// CHECK-WITHOUT-TESTENABLE: hw.module.extern @VeryGate(%I: i1, %E: i1) -> (O: i1)

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%clock: !seq.clock, %enable: i1, %test_enable: i1) {
  // CHECK-NOT: seq.clock_gate
  %cg0 = seq.clock_gate %clock, %enable
  %cg1 = seq.clock_gate %clock, %enable, %test_enable
  %cg2 = seq.clock_gate %clock, %enable sym @symbol

  // CHECK-DEFAULT: [[CAST:%.+]] = seq.from_clock %clock
  // CHECK-DEFAULT: hw.instance "ckg" @CKG(I: [[CAST]]: i1, E: %enable: i1, TE: %false: i1) -> (O: i1)
  // CHECK-DEFAULT: hw.instance "ckg" @CKG(I: [[CAST]]: i1, E: %enable: i1, TE: %test_enable: i1) -> (O: i1)
  // CHECK-DEFAULT: hw.instance "ckg" sym @symbol @CKG(I: [[CAST]]: i1, E: %enable: i1, TE: %false: i1) -> (O: i1)

  // CHECK-CUSTOM: [[CAST:%.+]] = seq.from_clock %clock
  // CHECK-CUSTOM: hw.instance "gated" @SuchClock(CI: [[CAST]]: i1, EN: %enable: i1, TEN: %false: i1) -> (CO: i1)
  // CHECK-CUSTOM: hw.instance "gated" @SuchClock(CI: [[CAST]]: i1, EN: %enable: i1, TEN: %test_enable: i1) -> (CO: i1)
  // CHECK-CUSTOM: hw.instance "gated" sym @symbol @SuchClock(CI: [[CAST]]: i1, EN: %enable: i1, TEN: %false: i1) -> (CO: i1)

  // CHECK-WITHOUT-TESTENABLE: [[CAST:%.+]] = seq.from_clock %clock
  // CHECK-WITHOUT-TESTENABLE: hw.instance "ckg" @VeryGate(I: [[CAST]]: i1, E: %enable: i1) -> (O: i1)
  // CHECK-WITHOUT-TESTENABLE: [[COMBINED_ENABLE:%.+]] = comb.or bin %enable, %test_enable : i1
  // CHECK-WITHOUT-TESTENABLE: hw.instance "ckg" @VeryGate(I: [[CAST]]: i1, E: [[COMBINED_ENABLE]]: i1) -> (O: i1)
  // CHECK-WITHOUT-TESTENABLE: hw.instance "ckg" sym @symbol @VeryGate(I: [[CAST]]: i1, E: %enable: i1) -> (O: i1)
}
