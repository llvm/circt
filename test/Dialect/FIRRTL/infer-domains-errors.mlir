// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains))' %s --verify-diagnostics --split-input-file

// Port annotated with same domain type twice.
firrtl.circuit "DomainCrossOnPort" {
  firrtl.domain @ClockDomain
  firrtl.module @DomainCrossOnPort(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    // expected-error @below {{illegal ClockDomain crossing in port #2}}
    // expected-note  @below {{1st instance: A}}
    // expected-note  @below {{2nd instance: B}}
    in %p: !firrtl.uint<1> domains [%A, %B]
  ) {}
}

// -----

// Illegal domain crossing - connect op.
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
    firrtl.connect %b, %a : !firrtl.uint<1>
  }
}

// -----

// Illegal domain crossing at matchingconnect op.
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

// -----

// Unable to infer domain of port, when port is driven by constant.
firrtl.circuit "UnableToInferDomainOfPortDrivenByConstant" {
  firrtl.domain @ClockDomain
  firrtl.module @Foo(in %i: !firrtl.uint<1>) {}

  firrtl.module @UnableToInferDomainOfPortDrivenByConstant() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // expected-error @below {{unable to infer value for domain port "ClockDomain"}}
    // expected-note  @below {{associated with hardware port "i"}}
    %foo_i = firrtl.instance foo @Foo(in i: !firrtl.uint<1>)
    firrtl.matchingconnect %foo_i, %c0_ui1 : !firrtl.uint<1>
  }
}

// -----

// Unable to infer domain of port, when port is driven by arithmetic on constant.
firrtl.circuit "UnableToInferDomainOfPortDrivenByConstantExpr" {
  firrtl.domain @ClockDomain
  firrtl.module @Foo(in %i: !firrtl.uint<2>) {}

  firrtl.module @UnableToInferDomainOfPortDrivenByConstantExpr() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.add %c0_ui1, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    // expected-error @below {{unable to infer value for domain port "ClockDomain"}}
    // expected-note  @below {{associated with hardware port "i"}}
    %foo_i = firrtl.instance foo @Foo(in i: !firrtl.uint<2>)
    firrtl.matchingconnect %foo_i, %0 : !firrtl.uint<2>
  }
}

// -----

// Incomplete extmodule domain information.

firrtl.circuit "IncompleteDomainInfoForExtModule" {
  firrtl.domain @ClockDomain

  firrtl.extmodule @Foo(in i: !firrtl.uint<1>)

  firrtl.module @IncompleteDomainInfoForExtModule(in %i: !firrtl.uint<1>) {
    // expected-error @below {{'firrtl.instance' op missing "ClockDomain" association for port "i"}}
    %foo_i = firrtl.instance foo @Foo(in i: !firrtl.uint<1>)
    firrtl.matchingconnect %foo_i, %i : !firrtl.uint<1>
  }
}
