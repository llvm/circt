// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=check}))' %s --verify-diagnostics --split-input-file

// CHECK-LABEL: IncompleteDomainInformation
firrtl.circuit "IncompleteDomainInformation" {
  firrtl.domain @ClockDomain

  firrtl.module private @Foo(
    // expected-error @below {{missing "ClockDomain" association for port "x"}}
    in  %x: !firrtl.uint<1>
  ) {}

  firrtl.module @IncompleteDomainInformation() {}
}
