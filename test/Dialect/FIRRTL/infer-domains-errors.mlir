// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains))' %s --verify-diagnostics --split-input-file

// Test case 1: Illegal domain crossing - both matchingconnect and connect should fail
firrtl.circuit "IllegalDomainCrossing" {
  firrtl.domain @ClockDomain {}
  firrtl.module @IllegalDomainCrossing(
    // expected-note@below {{operand is in domain defined here}}
    in %A: !firrtl.domain of @ClockDomain,
    // expected-note@below {{first operand is in domain defined here}}
    in %B: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>

    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Test case 2: Multiple domain crossings
firrtl.circuit "MultipleDomainCrossings" {
  firrtl.domain @ClockDomain {}
  firrtl.module @MultipleDomainCrossings(
    // expected-note@below {{operand is in domain defined here}}
    in %A: !firrtl.domain of @ClockDomain,
    // expected-note@below {{first operand is in domain defined here}}
    in %B: !firrtl.domain of @ClockDomain,
    // expected-note@below {{first operand is in domain defined here}}
    in %C: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%B],
    out %c: !firrtl.uint<1> domains [%C]
  ) {
    // expected-error@below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>

    // expected-error@below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %c, %a : !firrtl.uint<1>
  }
}

// -----

// Test case 3: Domain sequence mismatch - different lengths
firrtl.circuit "SequenceLengthMismatch" {
  firrtl.domain @ClockDomain {}
  firrtl.module @SequenceLengthMismatch(
    // expected-note@below {{operand (domain 1 of 2) is in domain defined here}}
    in %A: !firrtl.domain of @ClockDomain,
    // expected-note@below {{first operand is in domain defined here}}
    // expected-note@below {{operand (domain 2 of 2) is in domain defined here}}
    in %B: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A, %B],
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // expected-error@below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}
