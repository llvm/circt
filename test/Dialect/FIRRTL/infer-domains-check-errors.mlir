// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=check}))' %s --verify-diagnostics --split-input-file

// In check-mode, we check that the interface of public modules is fully annotated
// with domain inference
// CHECK-LABEL: MissingDomain
firrtl.circuit "MissingDomain" {
  firrtl.domain @ClockDomain

  firrtl.module @MissingDomain(
    // expected-error @below {{missing "ClockDomain" association for port "x"}}
    in %x: !firrtl.uint<1>
  ) {}
}

// CHECK-LABEL: MissingSecondDomain
firrtl.circuit "MissingSecondDomain" {
  firrtl.domain @ClockDomain
  firrtl.domain @PowerDomain

  firrtl.module @MissingSecondDomain(
    in %c : !firrtl.domain of @ClockDomain,
    // expected-error @below {{missing "PowerDomain" association for port "x"}}
    in %x : !firrtl.uint<1> domains [%c]
  ) {}
}

// CHECK-LABEL: UndrivenOutputDomain
firrtl.circuit "UndrivenOutputDomain" {
  firrtl.domain @ClockDomain

  firrtl.module @UndrivenOutputDomain(
    // expected-error @below {{unable to infer value for undriven domain port "c"}}
    out %c : !firrtl.domain of @ClockDomain
  ) {}
}

// CHECK-LABEL: UndrivenInstanceDomainPort
firrtl.circuit "UndrivenInstanceDomainPort" {
  firrtl.domain @ClockDomain

  firrtl.extmodule @Foo(in c : !firrtl.domain of @ClockDomain)

  firrtl.module @UndrivenInstanceDomainPort() {
    // expected-error @below {{unable to infer value for undriven domain port "c"}}
    %foo_c = firrtl.instance foo @Foo(in c : !firrtl.domain of @ClockDomain)
  }
}

// CHECK-LABEL: UndrivenInstanceChoiceDomainPort
firrtl.circuit "UndrivenInstanceChoiceDomainPort" {
  firrtl.domain @ClockDomain

  firrtl.option @Option {
    firrtl.option_case @X
  }

  firrtl.extmodule @Foo(in c : !firrtl.domain of @ClockDomain)
  firrtl.extmodule @Bar(in c : !firrtl.domain of @ClockDomain)

  firrtl.module @UndrivenInstanceChoiceDomainPort() {
    // expected-error @below {{unable to infer value for undriven domain port "c"}}
    %inst_c = firrtl.instance_choice inst @Foo alternatives @Option { @X -> @Bar } (in c : !firrtl.domain of @ClockDomain)
  }
}
