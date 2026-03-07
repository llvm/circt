// RUN: circt-opt %s -split-input-file -verify-diagnostics
// XFAIL: *

// Unable to determine domain type of domain source/destination.
//
// See: https://github.com/llvm/circt/issues/9398
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.domain @PowerDomain
  firrtl.module @Foo(
    in %in: !firrtl.domain<@ClockDomain>,
    out %out: !firrtl.domain<@PowerDomain>
  ) {
    %w = firrtl.wire : !firrtl.domain
    // expected-error @below {{could not determine domain-type of destination}}
    firrtl.domain.define %w, %in : !firrtl.domain<@ClockDomain>
    firrtl.domain.define %out, %w : !firrtl.domain<@ClockDomain>
  }
}
