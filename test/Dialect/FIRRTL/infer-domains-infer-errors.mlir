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
    in %c : !firrtl.domain<@ClockDomain()>,
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
    // expected-note @below {{input module port A declared here}}
    in %A: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{input module port B declared here}}
    in %B: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{a has domains [A : ClockDomain]}}
    in %a: !firrtl.uint<1> domains [%A],
    // expected-note @below {{b has domains [B : ClockDomain]}}
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// A value that mixes a colored operand with a colorless one is colored, not
// colorless, and still conflicts when used in a different domain.

// CHECK-LABEL: ColorlessPlusColoredStillError
firrtl.circuit "ColorlessPlusColoredStillError" {
  firrtl.domain @ClockDomain
  firrtl.module @ColorlessPlusColoredStillError(
    // expected-note @below {{input module port A declared here}}
    in %A: !firrtl.domain<@ClockDomain()>,
    in %x: !firrtl.uint<1> domains [%A],
    // expected-note @below {{input module port B declared here}}
    in %B: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{b has domains [B : ClockDomain]}}
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %y = firrtl.node %c1_ui1 : !firrtl.uint<1>
    // expected-note @below {{%0 has domains [A : ClockDomain]}}
    %0 = firrtl.eq %x, %y : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %0 : !firrtl.uint<1>
  }
}

// A public output port driven by a pure constant still requires an explicit
// domain association.

// CHECK-LABEL: PublicPortConstantStillMissing
firrtl.circuit "PublicPortConstantStillMissing" {
  firrtl.domain @ClockDomain
  // expected-note @below {{in module "PublicPortConstantStillMissing"}}
  firrtl.module @PublicPortConstantStillMissing(
    // expected-error @below {{missing "ClockDomain" association for port "a"}}
    out %a: !firrtl.uint<1>
  ) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.matchingconnect %a, %c1_ui1 : !firrtl.uint<1>
  }
}

// An undriven wire is colored (it is not colorless) and still errors when used
// in two different domains.

// CHECK-LABEL: UndrivenWireNotColorless
firrtl.circuit "UndrivenWireNotColorless" {
  firrtl.domain @ClockDomain
  firrtl.module @UndrivenWireNotColorless(
    // expected-note @below {{input module port A declared here}}
    in %A: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{input module port B declared here}}
    in %B: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{b has domains [B : ClockDomain]}}
    out %b: !firrtl.uint<1> domains [%B],
    out %a: !firrtl.uint<1> domains [%A]
  ) {
    // expected-note @below {{w has domains [A : ClockDomain]}}
    %w = firrtl.wire : !firrtl.uint<1>
    firrtl.matchingconnect %a, %w : !firrtl.uint<1>
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %w : !firrtl.uint<1>
  }
}

// Invalidvalue is not constant-like (it is a fresh unique value on every use)
// and must not be treated as colorless.

// CHECK-LABEL: InvalidValueNotColorless
firrtl.circuit "InvalidValueNotColorless" {
  firrtl.domain @ClockDomain
  firrtl.module @InvalidValueNotColorless(
    // expected-note @below {{input module port A declared here}}
    in %A: !firrtl.domain<@ClockDomain()>,
    out %a: !firrtl.uint<1> domains [%A],
    // expected-note @below {{input module port B declared here}}
    in %B: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{b has domains [B : ClockDomain]}}
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // expected-note @below {{%invalid_ui1 has domains [A : ClockDomain]}}
    %0 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.matchingconnect %a, %0 : !firrtl.uint<1>
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %0 : !firrtl.uint<1>
  }
}

// Cross-module colorless propagation is out of scope: a private module whose
// output is a purely-constant expression is _not_ treated as colorless at its
// instantiation sites, so this still errors.

// CHECK-LABEL: PrivateModuleConstantOutputStillErrors
firrtl.circuit "PrivateModuleConstantOutputStillErrors" {
  firrtl.domain @ClockDomain

  firrtl.module private @Bar(out %o: !firrtl.uint<4>) {
    %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
    %c2_ui4 = firrtl.constant 2 : !firrtl.uint<4>
    %0 = firrtl.add %c1_ui4, %c2_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
    %1 = firrtl.bits %0 3 to 0 : (!firrtl.uint<5>) -> !firrtl.uint<4>
    firrtl.matchingconnect %o, %1 : !firrtl.uint<4>
  }

  firrtl.module @PrivateModuleConstantOutputStillErrors(
    // expected-note @below {{input module port A declared here}}
    in %A: !firrtl.domain<@ClockDomain()>,
    out %a: !firrtl.uint<4> domains [%A],
    // expected-note @below {{input module port B declared here}}
    in %B: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{b has domains [B : ClockDomain]}}
    out %b: !firrtl.uint<4> domains [%B]
  ) {
    // expected-note @below {{bar.o has domains [A : ClockDomain]}}
    %bar_o = firrtl.instance bar @Bar(out o: !firrtl.uint<4>)
    firrtl.matchingconnect %a, %bar_o : !firrtl.uint<4>
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %bar_o : !firrtl.uint<4>
  }
}
