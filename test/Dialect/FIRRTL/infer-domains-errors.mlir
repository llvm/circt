// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains))' %s --verify-diagnostics --split-input-file

// Port annotated with same domain type twice.
firrtl.circuit "DomainCrossOnPort" {
  firrtl.domain @ClockDomain {}
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
  firrtl.domain @ClockDomain {}
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
  firrtl.domain @ClockDomain {}
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
