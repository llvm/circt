// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer-all}))' %s --verify-diagnostics

// A module-port association inferred from an internal connection is a summary,
// so the explanation should continue through the instance into that connection.
firrtl.circuit "Top" {
  firrtl.domain @ClockDomain

  firrtl.module @Child(
    in %D: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{input module port a is associated with input module port D}}
    in %a: !firrtl.uint<1> domains [%D],
    out %b: !firrtl.uint<1>
  ) {
    // expected-note @below {{domains of output module port b and input module port a are constrained to match by firrtl.matchingconnect}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }

  firrtl.module @Top(
    // expected-note @below {{input module port A declared here}}
    in %A: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{input module port B declared here}}
    in %B: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{b has domains [B : ClockDomain]}}
    out %b: !firrtl.uint<1> domains [%B]
  ) {
    // expected-note @below {{child.b has domains [A : ClockDomain]}}
    // expected-note @below {{domain inference path from output instance port child.b to input module port A}}
    // expected-note @below {{output instance port child.b is bound to output module port b}}
    // expected-note @below {{input instance port child.D is bound to input module port D}}
    %child_D, %child_a, %child_b = firrtl.instance child @Child(
      in D: !firrtl.domain<@ClockDomain()>,
      in a: !firrtl.uint<1> domains [D],
      out b: !firrtl.uint<1>
    )
    // expected-note @below {{input instance port child.D aliases input module port A}}
    firrtl.domain.define %child_D, %A : !firrtl.domain<@ClockDomain()>
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %child_b : !firrtl.uint<1>
  }
}
