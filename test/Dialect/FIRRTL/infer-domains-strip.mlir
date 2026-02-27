// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=strip}))' %s | FileCheck %s

firrtl.circuit "StripDomains" {
  firrtl.module @StripDomains() {}

  // CHECK-NOT: firrtl.domain @ClockDomain
  firrtl.domain @ClockDomain

  // CHECK-LABEL: firrtl.extmodule @Extmodule1(out x: !firrtl.uint<1>)
  firrtl.extmodule @Extmodule1(out D: !firrtl.domain of @ClockDomain, out x: !firrtl.uint<1> domains [D])

  // CHECK-LABEL: firrtl.extmodule @Extmodule2(out y: !firrtl.uint<1>)
  firrtl.extmodule @Extmodule2(out y: !firrtl.uint<1> domains [D], out D: !firrtl.domain of @ClockDomain)

  // CHECK-LABEL: firrtl.module @ExtmoduleUser(out %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
  // CHECK-NEXT:    %inst1_x = firrtl.instance inst1 @Extmodule1(out x: !firrtl.uint<1>)
  // CHECK-NEXT:    %inst2_y = firrtl.instance inst2 @Extmodule2(out y: !firrtl.uint<1>)
  // CHECK-NEXT:    firrtl.matchingconnect %x, %inst1_x : !firrtl.uint<1>
  // CHECK-NEXT:    firrtl.matchingconnect %y, %inst2_y : !firrtl.uint<1>
  // CHECK-NEXT:  }
  firrtl.module @ExtmoduleUser(out %x : !firrtl.uint<1>, out %y : !firrtl.uint<1>) {
    %inst1_D, %inst1_x = firrtl.instance inst1 @Extmodule1(out D: !firrtl.domain of @ClockDomain, out x: !firrtl.uint<1> domains [D])
    %inst2_y, %inst2_D = firrtl.instance inst2 @Extmodule2(out y: !firrtl.uint<1> domains [D], out D: !firrtl.domain of @ClockDomain)
    firrtl.matchingconnect %x, %inst1_x : !firrtl.uint<1>
    firrtl.matchingconnect %y, %inst2_y : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @StripWire() {
  // CHECK-NEXT:  }
  firrtl.module @StripWire(in %DI: !firrtl.domain of @ClockDomain, out %DO: !firrtl.domain of @ClockDomain) {
    %wire = firrtl.wire : !firrtl.domain
    firrtl.domain.define %wire, %DI
    firrtl.domain.define %DO, %wire
  }

  // CHECK-LABEL: firrtl.module @StripUnsafeDomainCast(
  // CHECK-SAME:    in  %i:  !firrtl.uint<1>,
  // CHECK-SAME:    out %o:  !firrtl.uint<1>) {
  // CHECK-NEXT:    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
  // CHECK-NEXT:  }
  firrtl.module @StripUnsafeDomainCast(
      in  %D1: !firrtl.domain of @ClockDomain,
      in  %D2: !firrtl.domain of @ClockDomain,
      in  %i:  !firrtl.uint<1> domains [%D1],
      out %o:  !firrtl.uint<1> domains [%D2]) {
    %0 = firrtl.unsafe_domain_cast %i domains %D2 : !firrtl.uint<1>
    firrtl.matchingconnect %o, %0 : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @StripAnonDomain() {
  // CHECK-NEXT:  }
  firrtl.module @StripAnonDomain(out %D: !firrtl.domain of @ClockDomain) {
    %0 = firrtl.domain.anon : !firrtl.domain of @ClockDomain
    firrtl.domain.define %D, %0
  }
}
