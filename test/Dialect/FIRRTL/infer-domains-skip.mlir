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
}
