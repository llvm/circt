// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{infer-public=true}))' %s --verify-diagnostics --split-input-file

// Port annotated with same domain type twice.
firrtl.circuit "DomainCrossOnPort" {
  firrtl.domain @ClockDomain
  firrtl.module @DomainCrossOnPort(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    // expected-error @below {{illegal "ClockDomain" crossing in port "p"}}
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

firrtl.circuit "Top" {
  firrtl.domain @ClockDomain

  // expected-error @below {{missing "ClockDomain" association for port "i"}}
  firrtl.extmodule @Top(in i: !firrtl.uint<1>)
}

// -----

// Conflicting extmodule domain information.

firrtl.circuit "Top" {
  firrtl.domain @ClockDomain

  firrtl.extmodule @Top(
    // expected-note @below {{associated with "ClockDomain" port "D1"}}
    in D1 : !firrtl.domain of @ClockDomain,
    // expected-note @below {{associated with "ClockDomain" port "D2"}}
    in D2 : !firrtl.domain of @ClockDomain,
    // expected-error @below {{ambiguous "ClockDomain" association for port "i"}}
    in i: !firrtl.uint<1> domains [D1, D2]
  )
}

// -----

// Domain not exported like it should be.

// -----

// Domain exported multiple times. Which do we choose?

firrtl.circuit "DoubleExportOfDomain" {
  firrtl.domain @ClockDomain

  firrtl.module @DoubleExportOfDomain(
    // expected-note @below {{candidate association "DI"}}
    in  %DI : !firrtl.domain of @ClockDomain,
    // expected-note @below {{candidate association "DO"}}
    out %DO : !firrtl.domain of @ClockDomain,
    in  %i  : !firrtl.uint<1> domains [%DO],
    // expected-error @below {{ambiguous "ClockDomain" association for port "o"}}
    out %o  : !firrtl.uint<1> domains []
  ) {
    // DI and DO are aliases
    firrtl.domain.define %DO, %DI

    // o is on same domain as i
    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
  }
}

