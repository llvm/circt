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

// Test that an EICG wrapper with domain ports is properly lowered.
// CHECK-LABEL: "FixupEICGWrapperDomains"
firrtl.circuit "FixupEICGWrapperDomains" {
  firrtl.domain @ClockDomain
  // CHECK-NOEICG: LegacyClockGateDomains
  // CHECK-EICG-NOT: LegacyClockGateDomains
  firrtl.extmodule @LegacyClockGateDomains(
    in in: !firrtl.clock domains [A],
    in test_en: !firrtl.uint<1> domains [A],
    in en: !firrtl.uint<1> domains [A],
    out out: !firrtl.clock domains [B],
    in A: !firrtl.domain<@ClockDomain()>,
    in B: !firrtl.domain<@ClockDomain()>
  ) attributes {defname = "EICG_wrapper"}

  // CHECK: FixupEICGWrapperDomains
  firrtl.module @FixupEICGWrapperDomains(
    in %clock: !firrtl.clock domains [%A],
    in %test_en: !firrtl.uint<1> domains [%A],
    in %en: !firrtl.uint<1> domains [%A],
    out %out: !firrtl.clock domains [%B],
    in %A: !firrtl.domain<@ClockDomain()>,
    in %B: !firrtl.domain<@ClockDomain()>
  ) {
    // CHECK-NOEICG: firrtl.instance
    // CHECK-EICG-NOT: firrtl.instance
    //
    // Domain ports become plain domain wires, created before any data wires.
    // CHECK-EICG-NEXT: %[[A:.+]] = firrtl.wire : !firrtl.domain<@ClockDomain()>
    // CHECK-EICG-NEXT: %[[B:.+]] = firrtl.wire : !firrtl.domain<@ClockDomain()>
    //
    // Data port wires preserve their original domain associations.
    // CHECK-EICG-NEXT: %[[CLK:.+]] = firrtl.wire domains[%[[A]]]
    // CHECK-EICG-NEXT: %[[TEST_EN:.+]] = firrtl.wire domains[%[[A]]]
    // CHECK-EICG-NEXT: %[[EN:.+]] = firrtl.wire domains[%[[A]]]
    // CHECK-EICG-NEXT: %[[OUT:.+]] = firrtl.wire domains[%[[B]]]
    //
    // Each input is unsafe-cast into the anonymous domain before reaching
    // the intrinsic.
    // CHECK-EICG-NEXT: %[[ANON:.+]] = firrtl.domain.anon : !firrtl.domain<@ClockDomain()>
    // CHECK-EICG-NEXT: %[[CLK_C:.+]] = firrtl.unsafe_domain_cast %[[CLK]] domains[%[[ANON]]]
    // CHECK-EICG-NEXT: %[[TEST_EN_C:.+]] = firrtl.unsafe_domain_cast %[[TEST_EN]] domains[%[[ANON]]]
    // CHECK-EICG-NEXT: %[[EN_C:.+]] = firrtl.unsafe_domain_cast %[[EN]] domains[%[[ANON]]]
    //
    // The intrinsic uses the cast inputs with en/test_en swapped.
    // CHECK-EICG: %[[INT:.+]] = firrtl.int.generic "circt_clock_gate" %[[CLK_C]], %[[EN_C]], %[[TEST_EN_C]]
    //
    // The result is unsafe-cast back to the output's real domain and
    // connected to the output wire.
    // CHECK-EICG: %[[OUT_C:.+]] = firrtl.unsafe_domain_cast %[[INT]] domains[%[[B]]]
    // CHECK-EICG: firrtl.matchingconnect %[[OUT]], %[[OUT_C]]
    %ckg_in, %ckg_test_en, %ckg_en, %ckg_out, %ckg_A, %ckg_B = firrtl.instance ckg @LegacyClockGateDomains(
      in in: !firrtl.clock domains [A],
      in test_en: !firrtl.uint<1> domains [A],
      in en: !firrtl.uint<1> domains [A],
      out out: !firrtl.clock domains [B],
      in A: !firrtl.domain<@ClockDomain()>,
      in B: !firrtl.domain<@ClockDomain()>
    )
    // All uses are properly RAUW'd.
    // CHECK-EICG-NEXT: firrtl.domain.define %[[A]], %A
    // CHECK-EICG-NEXT: firrtl.domain.define %[[B]], %B
    // CHECK-EICG-NEXT: firrtl.matchingconnect %[[CLK]], %clock
    // CHECK-EICG-NEXT: firrtl.matchingconnect %[[TEST_EN]], %test_en
    // CHECK-EICG-NEXT: firrtl.matchingconnect %[[EN]], %en
    // CHECK-EICG-NEXT: firrtl.matchingconnect %out, %[[OUT]]
    firrtl.domain.define %ckg_A, %A : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %ckg_B, %B : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %ckg_in, %clock : !firrtl.clock
    firrtl.matchingconnect %ckg_test_en, %test_en : !firrtl.uint<1>
    firrtl.matchingconnect %ckg_en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %out, %ckg_out : !firrtl.clock
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

