// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer}))' %s --verify-diagnostics

// in "infer" mode, infer-domains requires that the interfaces of public
// modules are fully annotated with domain associations, but will still
// perform domain inference on the body of a public module, and will do full
// inference for private modules.

// This test suite checks for errors which do not occur when the mode is
// infer-all.

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

// Test that domain crossing errors are still caught when in infer mode.
// Catching this involves processing the module without writing back to the IR.
firrtl.circuit "IllegalDomainCrossing" {
  firrtl.domain @ClockDomain
  firrtl.module @IllegalDomainCrossing(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    // expected-note @below {{2nd operand has domains: [ClockDomain: A]}}
    in %a: !firrtl.uint<1> domains [%A],
    // expected-note @below {{1st operand has domains: [ClockDomain: B]}}
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
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

// Unable to infer domain of port, when port is driven by constant.
firrtl.circuit "UnableToInferDomainOfPortDrivenByConstant" {
  firrtl.domain @ClockDomain
  firrtl.module private @Foo(in %i: !firrtl.uint<1>) {}

  firrtl.module @UnableToInferDomainOfPortDrivenByConstant() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // expected-error @below {{unable to infer value for undriven domain port "ClockDomain"}}
    // expected-note  @below {{associated with hardware port "i"}}
    %foo_i = firrtl.instance foo @Foo(in i: !firrtl.uint<1>)
    firrtl.matchingconnect %foo_i, %c0_ui1 : !firrtl.uint<1>
  }
}

// Unable to infer domain of port, when port is driven by arithmetic on constant.
firrtl.circuit "UnableToInferDomainOfPortDrivenByConstantExpr" {
  firrtl.domain @ClockDomain
  firrtl.module private @Foo(in %i: !firrtl.uint<2>) {}

  firrtl.module @UnableToInferDomainOfPortDrivenByConstantExpr() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.add %c0_ui1, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    // expected-error @below {{unable to infer value for undriven domain port "ClockDomain"}}
    // expected-note  @below {{associated with hardware port "i"}}
    %foo_i = firrtl.instance foo @Foo(in i: !firrtl.uint<2>)
    firrtl.matchingconnect %foo_i, %0 : !firrtl.uint<2>
  }
}

