// RUN: circt-opt %s --externalize-clock-gate="name=SuchClock input=CI output=CO enable=EN test-enable=TEN instance-name=gated" --verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --externalize-clock-gate="name=VeryGate test-enable=" --verify-diagnostics | FileCheck %s --check-prefixes=CHECK,CHECK-WITHOUT-TESTENABLE

// RUN: circt-opt --hw-externalize="op=seq.clock_gate module-name=SuchClock instance-name=gated port-names=input=CI,output=CO,enable=EN,test-enable=TEN" %s | FileCheck %s

// RUN: circt-opt --hw-externalize="op=seq.clock_gate module-name=VeryGate instance-name=gated port-names=input=CI,output=CO,enable=EN" %s | FileCheck %s --check-prefix=WITHOUT-TESTENABLE

// CHECK: hw.module.extern @SuchClock(%CI: i1, %EN: i1, %TEN: i1) -> (CO: i1)
// CHECK-WITHOUT-TESTENABLE: hw.module.extern @VeryGate(%I: i1, %E: i1) -> (O: i1)

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%clock: i1, %enable: i1, %test_enable: i1) {
  // CHECK-NOT: seq.clock_gate
  %cg0 = seq.clock_gate %clock, %enable
  %cg1 = seq.clock_gate %clock, %enable, %test_enable

  // CHECK: hw.instance "gated" @SuchClock(CI: %clock: i1, EN: %enable: i1, TEN: %false: i1) -> (CO: i1)
  // CHECK: hw.instance "gated" @SuchClock(CI: %clock: i1, EN: %enable: i1, TEN: %test_enable: i1) -> (CO: i1)

  // CHECK-WITHOUT-TESTENABLE: hw.instance "ckg" @VeryGate(I: %clock: i1, E: %enable: i1) -> (O: i1)
  // CHECK-WITHOUT-TESTENABLE: [[COMBINED_ENABLE:%.+]] = comb.or bin %enable, %test_enable : i1
  // CHECK-WITHOUT-TESTENABLE: hw.instance "ckg" @VeryGate(I: %clock: i1, E: [[COMBINED_ENABLE]]: i1) -> (O: i1)
}
