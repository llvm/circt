// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer-all}))' %s --verify-diagnostics

// An alias which is present in only one InstanceChoiceOp candidate must not
// be imported into the parent.
firrtl.circuit "NonCommonChoiceAlias" {
  firrtl.domain @ClockDomain
  firrtl.option @Choice {
    firrtl.option_case @A
    firrtl.option_case @B
  }

  firrtl.module @ChoiceA(
    // expected-note @below {{input module port A declared here}}
    in %A: !firrtl.domain<@ClockDomain()>,
    in %B: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{output module port O declared here}}
    out %O: !firrtl.domain<@ClockDomain()>
  ) {
    // expected-note @below {{output module port O aliases input module port A}}
    firrtl.domain.define %O, %A : !firrtl.domain<@ClockDomain()>
  }
  firrtl.module @ChoiceB(
    in %A: !firrtl.domain<@ClockDomain()>,
    in %B: !firrtl.domain<@ClockDomain()>,
    out %O: !firrtl.domain<@ClockDomain()>
  ) {
    firrtl.domain.define %O, %B : !firrtl.domain<@ClockDomain()>
  }

  firrtl.module @NonCommonChoiceAlias(
    // expected-note @below {{input module port D1 declared here}}
    in %D1: !firrtl.domain<@ClockDomain()>,
    in %D2: !firrtl.domain<@ClockDomain()>
  ) {
    // expected-note @below {{output instance port choice.O declared here}}
    // expected-note @below {{input instance port choice.A declared here}}
    %A, %B, %O = firrtl.instance_choice choice @ChoiceA alternatives @Choice {
      @A -> @ChoiceB, @B -> @ChoiceB
    } (
      in A: !firrtl.domain<@ClockDomain()>,
      in B: !firrtl.domain<@ClockDomain()>,
      out O: !firrtl.domain<@ClockDomain()>)
    // expected-note @below {{input instance port choice.A aliases input module port D1}}
    firrtl.domain.define %A, %D1 : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %B, %D2 : !firrtl.domain<@ClockDomain()>
    %X = firrtl.wire : !firrtl.domain<@ClockDomain()>
    // expected-note @below {{X aliases output instance port choice.O}}
    firrtl.domain.define %X, %O : !firrtl.domain<@ClockDomain()>
    // expected-note @below {{x has domains [choice.O : ClockDomain]}}
    // expected-note @below {{domain inference path from x to output instance port choice.O}}
    // expected-note @below {{x is associated with X}}
    %x = firrtl.wire domains[%X] : !firrtl.uint<1>
        domains[!firrtl.domain<@ClockDomain()>]
    // expected-note @below {{y has domains [D1 : ClockDomain]}}
    %y = firrtl.wire domains[%D1] : !firrtl.uint<1>
        domains[!firrtl.domain<@ClockDomain()>]
    // expected-error @+1 {{illegal domain crossing}}
    firrtl.matchingconnect %x, %y : !firrtl.uint<1>
  }
}
