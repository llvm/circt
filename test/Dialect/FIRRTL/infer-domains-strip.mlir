// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=strip}))' %s | FileCheck %s

firrtl.circuit "StripDomains" {
  firrtl.module @StripDomains() {}

  // CHECK-NOT: firrtl.domain @ClockDomain
  firrtl.domain @ClockDomain

  // CHECK-NOT: firrtl.domain @PowerDomain
  firrtl.domain @PowerDomain [
    #firrtl.domain.field<"name", !firrtl.string>,
    #firrtl.domain.field<"voltage", !firrtl.integer>
  ]

  // CHECK-LABEL: firrtl.extmodule @Extmodule1(out x: !firrtl.uint<1>)
  firrtl.extmodule @Extmodule1(out D: !firrtl.domain<@ClockDomain()>, out x: !firrtl.uint<1> domains [D])

  // CHECK-LABEL: firrtl.extmodule @Extmodule2(out y: !firrtl.uint<1>)
  firrtl.extmodule @Extmodule2(out y: !firrtl.uint<1> domains [D], out D: !firrtl.domain<@ClockDomain()>)

  // CHECK-LABEL: firrtl.module @ExtmoduleUser(out %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
  // CHECK-NEXT:    %inst1_x = firrtl.instance inst1 @Extmodule1(out x: !firrtl.uint<1>)
  // CHECK-NEXT:    %inst2_y = firrtl.instance inst2 @Extmodule2(out y: !firrtl.uint<1>)
  // CHECK-NEXT:    firrtl.matchingconnect %x, %inst1_x : !firrtl.uint<1>
  // CHECK-NEXT:    firrtl.matchingconnect %y, %inst2_y : !firrtl.uint<1>
  // CHECK-NEXT:  }
  firrtl.module @ExtmoduleUser(out %x : !firrtl.uint<1>, out %y : !firrtl.uint<1>) {
    %inst1_D, %inst1_x = firrtl.instance inst1 @Extmodule1(out D: !firrtl.domain<@ClockDomain()>, out x: !firrtl.uint<1> domains [D])
    %inst2_y, %inst2_D = firrtl.instance inst2 @Extmodule2(out y: !firrtl.uint<1> domains [D], out D: !firrtl.domain<@ClockDomain()>)
    firrtl.matchingconnect %x, %inst1_x : !firrtl.uint<1>
    firrtl.matchingconnect %y, %inst2_y : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @StripWire() {
  // CHECK-NEXT:  }
  firrtl.module @StripWire(in %DI: !firrtl.domain<@ClockDomain()>, out %DO: !firrtl.domain<@ClockDomain()>) {
    %wire = firrtl.wire : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %wire, %DI : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %DO, %wire : !firrtl.domain<@ClockDomain()>
  }

  // CHECK-LABEL: firrtl.module @StripUnsafeDomainCast(
  // CHECK-SAME:    in  %i:  !firrtl.uint<1>,
  // CHECK-SAME:    out %o:  !firrtl.uint<1>) {
  // CHECK-NEXT:    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
  // CHECK-NEXT:  }
  firrtl.module @StripUnsafeDomainCast(
      in  %D1: !firrtl.domain<@ClockDomain()>,
      in  %D2: !firrtl.domain<@ClockDomain()>,
      in  %i:  !firrtl.uint<1> domains [%D1],
      out %o:  !firrtl.uint<1> domains [%D2]) {
    %0 = firrtl.unsafe_domain_cast %i domains[%D2] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %o, %0 : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @StripAnonDomain() {
  // CHECK-NEXT:  }
  firrtl.module @StripAnonDomain(out %D: !firrtl.domain<@ClockDomain()>) {
    %0 = firrtl.domain.anon : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %D, %0 : !firrtl.domain<@ClockDomain()>
  }

  // CHECK-LABEL: firrtl.module @StripNamedDomain() {
  // CHECK-NEXT:  }
  firrtl.module @StripNamedDomain(out %D: !firrtl.domain<@ClockDomain()>) {
    %my_domain = firrtl.domain.create : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %D, %my_domain : !firrtl.domain<@ClockDomain()>
  }

  // CHECK-LABEL: firrtl.module @StripDomainSubfield() {
  // CHECK-NEXT:    %[[NAME:.+]] = firrtl.string "VDD"
  // CHECK-NEXT:    %[[VOLTAGE:.+]] = firrtl.integer 1800
  // CHECK-NEXT:  }
  firrtl.module @StripDomainSubfield(out %D: !firrtl.domain<@PowerDomain(name: !firrtl.string, voltage: !firrtl.integer)>) {
    %name = firrtl.string "VDD"
    %voltage = firrtl.integer 1800
    %my_domain = firrtl.domain.create(%name, %voltage) : !firrtl.domain<@PowerDomain(name: !firrtl.string, voltage: !firrtl.integer)>
    %extracted_name = firrtl.domain.subfield %my_domain[name] : !firrtl.domain<@PowerDomain(name: !firrtl.string, voltage: !firrtl.integer)>
    %extracted_voltage = firrtl.domain.subfield %my_domain[voltage] : !firrtl.domain<@PowerDomain(name: !firrtl.string, voltage: !firrtl.integer)>
    firrtl.domain.define %D, %my_domain : !firrtl.domain<@PowerDomain(name: !firrtl.string, voltage: !firrtl.integer)>
  }

  // Test that domain subfield with non-domain users is replaced with unknown value.
  // CHECK-LABEL: firrtl.module @StripDomainSubfieldWithUse(out %x: !firrtl.integer) {
  // CHECK-NEXT:    %[[UNKNOWN:.+]] = firrtl.unknown : !firrtl.integer
  // CHECK-NEXT:    firrtl.propassign %x, %[[UNKNOWN]] : !firrtl.integer
  // CHECK-NEXT:  }
  firrtl.module @StripDomainSubfieldWithUse(
      in %D: !firrtl.domain<@PowerDomain(name: !firrtl.string, voltage: !firrtl.integer)>,
      out %x: !firrtl.integer) {
    %voltage = firrtl.domain.subfield %D[voltage] : !firrtl.domain<@PowerDomain(name: !firrtl.string, voltage: !firrtl.integer)>
    firrtl.propassign %x, %voltage : !firrtl.integer
  }

  // CHECK-LABEL: firrtl.module @StripWireDomains(
  // CHECK-SAME:    in %a: !firrtl.uint<1>) {
  // CHECK-NEXT:    %w = firrtl.wire : !firrtl.uint<1>
  // CHECK-NEXT:    firrtl.matchingconnect %w, %a : !firrtl.uint<1>
  // CHECK-NEXT:  }
  firrtl.module @StripWireDomains(
    in %D: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<1> domains [%D]
  ) {
    %w = firrtl.wire domains[%D] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %w, %a : !firrtl.uint<1>
  }
}
