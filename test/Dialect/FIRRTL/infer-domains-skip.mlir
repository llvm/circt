// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=check skip-domain=PowerDomain}))' %s | FileCheck %s --implicit-check-not=PowerDomain

// Verify skip-domain strips PowerDomain from the IR while preserving ClockDomain.

// CHECK-LABEL: firrtl.circuit "SkipDomains"
// CHECK: firrtl.domain @ClockDomain
firrtl.circuit "SkipDomains" {
  firrtl.domain @PowerDomain
  firrtl.domain @ClockDomain

  // CHECK-LABEL: firrtl.module @SkipDomains()
  firrtl.module @SkipDomains() {}

  // CHECK-LABEL: firrtl.extmodule @Ext
  // CHECK-SAME: out CK: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME: out data: !firrtl.uint<8> domains [CK]
  firrtl.extmodule @Ext(
    out CK: !firrtl.domain<@ClockDomain()>,
    out PW: !firrtl.domain<@PowerDomain()>,
    out data: !firrtl.uint<8> domains [CK, PW]
  )

  // CHECK-LABEL: firrtl.module @Main
  // CHECK-SAME: in %CK: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME: out %x: !firrtl.uint<8> domains [%CK]
  // CHECK: %{{.+}} = firrtl.instance ext @Ext
  // CHECK: %w = firrtl.wire domains[%{{.+}}] : !firrtl.uint<8>
  firrtl.module @Main(
      in %CK: !firrtl.domain<@ClockDomain()>,
      out %x: !firrtl.uint<8> domains [%CK]) {
    %ext_CK, %ext_PW, %ext_data = firrtl.instance ext @Ext(
      out CK: !firrtl.domain<@ClockDomain()>,
      out PW: !firrtl.domain<@PowerDomain()>,
      out data: !firrtl.uint<8> domains [CK, PW]
    )
    %w = firrtl.wire domains[%ext_CK, %ext_PW] : !firrtl.uint<8> domains[!firrtl.domain<@ClockDomain()>, !firrtl.domain<@PowerDomain()>]
    firrtl.matchingconnect %w, %ext_data : !firrtl.uint<8>
    %cast = firrtl.unsafe_domain_cast %w domains[%CK] : !firrtl.uint<8> domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %x, %cast : !firrtl.uint<8>
  }

  // A cast with a skipped and a kept domain keeps the cast to the kept domain.
  // CHECK-LABEL: firrtl.module @MixedCast
  // CHECK-SAME: in %CA: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME: in %CB: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME: in %x: !firrtl.uint<1> domains [%CA]
  // CHECK-SAME: out %y: !firrtl.uint<1> domains [%CB]
  // CHECK: %[[CAST:.+]] = firrtl.unsafe_domain_cast %x domains[%CB] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>]
  // CHECK: firrtl.matchingconnect %y, %[[CAST]] : !firrtl.uint<1>
  firrtl.module @MixedCast(
      in %CA: !firrtl.domain<@ClockDomain()>,
      in %CB: !firrtl.domain<@ClockDomain()>,
      in %P1: !firrtl.domain<@PowerDomain()>,
      in %P2: !firrtl.domain<@PowerDomain()>,
      in %x: !firrtl.uint<1> domains [%CA, %P1],
      out %y: !firrtl.uint<1> domains [%CB, %P2]) {
    %cast = firrtl.unsafe_domain_cast %x domains[%CB, %P2] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>, !firrtl.domain<@PowerDomain()>]
    firrtl.matchingconnect %y, %cast : !firrtl.uint<1>
  }
}
