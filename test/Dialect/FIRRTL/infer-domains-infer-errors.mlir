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

  // expected-note @below {{in module "MissingDomain"}}
  firrtl.module @MissingDomain(
    // expected-error @below {{missing "ClockDomain" association for port "x"}}
    in %x: !firrtl.uint<1>
  ) {}
}

// CHECK-LABEL: MissingSecondDomain
firrtl.circuit "MissingSecondDomain" {
  firrtl.domain @ClockDomain
  firrtl.domain @PowerDomain

  // expected-note @below {{in module "MissingSecondDomain"}}
  firrtl.module @MissingSecondDomain(
    in %c : !firrtl.domain of @ClockDomain,
    // expected-error @below {{missing "PowerDomain" association for port "x"}}
    in %x : !firrtl.uint<1> domains [%c]
  ) {}
}

// Test that domain crossing errors are still caught when in infer mode.
// Catching this involves processing the module without writing back to the IR.

// CHECK-LABEL: IllegalDomainCrossing
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
