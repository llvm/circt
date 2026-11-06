// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-strip-domains))' --verify-diagnostics %s

firrtl.circuit "StripDomainErrors" {
  firrtl.module @StripDomainErrors() {
    // expected-error @below {{'builtin.unrealized_conversion_cast' op cannot be stripped}}
    %0 = builtin.unrealized_conversion_cast to !firrtl.domain<@ClockDomain()>
  }
}
