// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=check skip-domain=PowerDomain}))' %s | FileCheck %s

// Verify skip-domain strips PowerDomain from the IR while preserving ClockDomain.

// CHECK-LABEL: firrtl.circuit "SkipDomains"
// CHECK-NOT: firrtl.domain @PowerDomain
// CHECK: firrtl.domain @ClockDomain
firrtl.circuit "SkipDomains" {
  firrtl.domain @PowerDomain
  firrtl.domain @ClockDomain

  // CHECK-LABEL: firrtl.module @SkipDomains()
  firrtl.module @SkipDomains() {}

  // CHECK-LABEL: firrtl.extmodule @Ext(out CK: !firrtl.domain<@ClockDomain()>, out data: !firrtl.uint<8> domains [CK])
  firrtl.extmodule @Ext(
    out CK: !firrtl.domain<@ClockDomain()>,
    out PW: !firrtl.domain<@PowerDomain()>,
    out data: !firrtl.uint<8> domains [CK, PW]
  )

  // CHECK-LABEL: firrtl.module @Main(in %CK: !firrtl.domain<@ClockDomain()>, out %x: !firrtl.uint<8> domains [%CK]) {
  // CHECK-NEXT:    %ext_CK, %ext_data = firrtl.instance ext @Ext(out CK: !firrtl.domain<@ClockDomain()>, out data: !firrtl.uint<8> domains [CK])
  // CHECK-NEXT:    %w = firrtl.wire domains[%ext_CK] : !firrtl.uint<8>
  // CHECK-NEXT:    firrtl.matchingconnect %w, %ext_data : !firrtl.uint<8>
  // CHECK-NEXT:    %{{.+}} = firrtl.unsafe_domain_cast %w domains[%CK] : !firrtl.uint<8>
  // CHECK-NEXT:    firrtl.matchingconnect %x, %{{.+}} : !firrtl.uint<8>
  // CHECK-NEXT:  }
  firrtl.module @Main(
      in %CK: !firrtl.domain<@ClockDomain()>,
      out %x: !firrtl.uint<8> domains [%CK]) {
    %ext_CK, %ext_PW, %ext_data = firrtl.instance ext @Ext(
      out CK: !firrtl.domain<@ClockDomain()>,
      out PW: !firrtl.domain<@PowerDomain()>,
      out data: !firrtl.uint<8> domains [CK, PW]
    )
    // After PowerDomain is stripped, only ClockDomain remains on the wire.
    %w = firrtl.wire domains[%ext_CK, %ext_PW] : !firrtl.uint<8> domains[!firrtl.domain<@ClockDomain()>, !firrtl.domain<@PowerDomain()>]
    firrtl.matchingconnect %w, %ext_data : !firrtl.uint<8>
    // Cast across clock domains so the remaining connection is legal for checking.
    %cast = firrtl.unsafe_domain_cast %w domains[%CK] : !firrtl.uint<8> domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %x, %cast : !firrtl.uint<8>
  }

  // CHECK-LABEL: firrtl.module @DomainOps(out %CK: !firrtl.domain<@ClockDomain()>) {
  // CHECK-NEXT:    %ck_anon = firrtl.domain.anon : !firrtl.domain<@ClockDomain()>
  // CHECK-NEXT:    firrtl.domain.define %CK, %ck_anon : !firrtl.domain<@ClockDomain()>
  // CHECK-NEXT:  }
  firrtl.module @DomainOps(
      out %CK: !firrtl.domain<@ClockDomain()>,
      out %PW: !firrtl.domain<@PowerDomain()>) {
    %ck_anon = firrtl.domain.anon : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %CK, %ck_anon : !firrtl.domain<@ClockDomain()>
    %pw_anon = firrtl.domain.anon : !firrtl.domain<@PowerDomain()>
    firrtl.domain.define %PW, %pw_anon : !firrtl.domain<@PowerDomain()>
  }

  // CHECK-LABEL: firrtl.module @MixedDomainWire(in %CK: !firrtl.domain<@ClockDomain()>, in %a: !firrtl.uint<1> domains [%CK]) {
  // CHECK-NEXT:    %w = firrtl.wire domains[%CK] : !firrtl.uint<1>
  // CHECK-NEXT:    firrtl.matchingconnect %w, %a : !firrtl.uint<1>
  // CHECK-NEXT:  }
  firrtl.module @MixedDomainWire(
      in %CK: !firrtl.domain<@ClockDomain()>,
      in %PW: !firrtl.domain<@PowerDomain()>,
      in %a: !firrtl.uint<1> domains [%CK, %PW]) {
    %w = firrtl.wire domains[%CK, %PW] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>, !firrtl.domain<@PowerDomain()>]
    firrtl.matchingconnect %w, %a : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @UnsafeCast(in %CK: !firrtl.domain<@ClockDomain()>, in %a: !firrtl.uint<1> domains [%CK], out %b: !firrtl.uint<1> domains [%CK]) {
  // CHECK-NEXT:    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  // CHECK-NEXT:  }
  firrtl.module @UnsafeCast(
      in %CK: !firrtl.domain<@ClockDomain()>,
      in %PW: !firrtl.domain<@PowerDomain()>,
      in %a: !firrtl.uint<1> domains [%CK],
      out %b: !firrtl.uint<1> domains [%CK, %PW]) {
    %0 = firrtl.unsafe_domain_cast %a domains[%PW] : !firrtl.uint<1> domains[!firrtl.domain<@PowerDomain()>]
    firrtl.matchingconnect %b, %0 : !firrtl.uint<1>
  }
}
