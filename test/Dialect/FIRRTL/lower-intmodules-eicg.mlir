// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intmodules))' --split-input-file %s | FileCheck %s --check-prefixes=CHECK,CHECK-NOEICG
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intmodules{fixup-eicg-wrapper}))' --split-input-file %s  --verify-diagnostics | FileCheck %s --check-prefixes=CHECK,CHECK-EICG

// CHECK-LABEL: "FixupEICGWrapper"
firrtl.circuit "FixupEICGWrapper" {
  // CHECK-NOEICG: LegacyClockGate
  // CHECK-EICG-NOT: LegacyClockGate
  // expected-warning @below {{Annotation firrtl.transforms.DedupGroupAnnotation on EICG_wrapper is dropped}}
  firrtl.extmodule @LegacyClockGate(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {
    defname = "EICG_wrapper",
    annotations = [{class = "firrtl.transforms.DedupGroupAnnotation", group = "foo"}]}

  // CHECK: FixupEICGWrapper
  firrtl.module @FixupEICGWrapper(in %clock: !firrtl.clock, in %test_en: !firrtl.uint<1>, in %en: !firrtl.uint<1>) {
    // CHECK-NOEICG: firrtl.instance
    // CHECK-EICG-NOT: firrtl.instance
    // CHECK-EICG-DAG: firrtl.matchingconnect %[[CLK:.+]], %clock : !firrtl.clock
    // CHECK-EICG-DAG: firrtl.matchingconnect %[[TEST_EN:.+]], %test_en : !firrtl.uint<1>
    // CHECK-EICG-DAG: firrtl.matchingconnect %[[EN:.+]], %en : !firrtl.uint<1>
    // CHECK-EICG-DAG: %[[CLK]] = firrtl.wire : !firrtl.clock
    // CHECK-EICG-DAG: %[[TEST_EN]] = firrtl.wire : !firrtl.uint<1>
    // CHECK-EICG-DAG: %[[EN]] = firrtl.wire : !firrtl.uint<1>
    // CHECK-EICG-DAG: %3 = firrtl.int.generic "circt_clock_gate"  %[[CLK]], %[[EN]], %[[TEST_EN]] : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.clock
    %ckg_in, %ckg_test_en, %ckg_en, %ckg_out = firrtl.instance ckg @LegacyClockGate(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.matchingconnect %ckg_in, %clock : !firrtl.clock
    firrtl.matchingconnect %ckg_test_en, %test_en : !firrtl.uint<1>
    firrtl.matchingconnect %ckg_en, %en : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "EICGWrapperPortName" {
  firrtl.extmodule @BadClockGate(in in: !firrtl.clock,
                                 // expected-error @below {{expected port named 'test_en'}}
                                 in en: !firrtl.uint<1>,
                                 in test_en: !firrtl.uint<1>,
                                 out out: !firrtl.clock)
    attributes { defname = "EICG_wrapper" }

  firrtl.module @EICGWrapperPortName(in %clock: !firrtl.clock, in %test_en: !firrtl.uint<1>, in %en: !firrtl.uint<1>) {
    %ckg_in, %ckg_en, %ckg_test_en, %ckg_out = firrtl.instance ckg @BadClockGate(in in: !firrtl.clock, in en: !firrtl.uint<1>, in test_en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.matchingconnect %ckg_in, %clock : !firrtl.clock
    firrtl.matchingconnect %ckg_test_en, %test_en : !firrtl.uint<1>
    firrtl.matchingconnect %ckg_en, %en : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: "FixupEICGWrapper2"
firrtl.circuit "FixupEICGWrapper2" {
  // CHECK-NOEICG: LegacyClockGateNoTestEn
  // CHECK-EICG-NOT: LegacyClockGateNoTestEn
  firrtl.extmodule @LegacyClockGateNoTestEn(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  // CHECK: FixupEICGWrapper2
  firrtl.module @FixupEICGWrapper2(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK-NOEICG: firrtl.instance
    // CHECK-EICG-NOT: firrtl.instance
    // CHECK-EICG: firrtl.int.generic "circt_clock_gate"
    %ckg_in, %ckg_en, %ckg_out = firrtl.instance ckg @LegacyClockGateNoTestEn(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.matchingconnect %ckg_in, %clock : !firrtl.clock
    firrtl.matchingconnect %ckg_en, %en : !firrtl.uint<1>
  }
}

