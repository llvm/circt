// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains{mode=infer-all}))' %s --verify-diagnostics

// Port annotated with same domain type twice.
firrtl.circuit "DomainCrossOnPort" {
  firrtl.domain @ClockDomain
  // expected-note @below {{in module "DomainCrossOnPort"}}
  firrtl.module @DomainCrossOnPort(
    // expected-note @below {{associated with "ClockDomain" port "A"}}
    in %A: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{associated with "ClockDomain" port "B"}}
    in %B: !firrtl.domain<@ClockDomain()>,
    // expected-error @below {{duplicate "ClockDomain" association for port "p"}}
    in %p: !firrtl.uint<1> domains [%A, %B]
  ) {}
}

// Illegal domain crossing via connect op.
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
    firrtl.connect %b, %a : !firrtl.uint<1>
  }
}

// Illegal domain crossing at matchingconnect op.
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

// Incomplete extmodule domain information.
firrtl.circuit "Top" {
  firrtl.domain @ClockDomain

  // expected-note  @+2 {{in extmodule "Top"}}
  // expected-error @+1 {{missing "ClockDomain" association for port "i"}}
  firrtl.extmodule @Top(in i: !firrtl.uint<1>)
}

// Conflicting extmodule domain information.
firrtl.circuit "Top" {
  firrtl.domain @ClockDomain

  // expected-note @below {{in extmodule "Top"}}
  firrtl.extmodule @Top(
    // expected-note @below {{associated with "ClockDomain" port "D1"}}
    in D1 : !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{associated with "ClockDomain" port "D2"}}
    in D2 : !firrtl.domain<@ClockDomain()>,
    // expected-error @below {{duplicate "ClockDomain" association for port "i"}}
    in i: !firrtl.uint<1> domains [D1, D2]
  )
}

// Domain exported multiple times. Which do we choose?
firrtl.circuit "DoubleExportOfDomain" {
  firrtl.domain @ClockDomain

  // expected-note @below {{in module "DoubleExportOfDomain"}}
  firrtl.module @DoubleExportOfDomain(
    // expected-note @below {{candidate association "DI"}}
    in  %DI : !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{candidate association "DO"}}
    out %DO : !firrtl.domain<@ClockDomain()>,
    in  %i  : !firrtl.uint<1> domains [%DO],
    // expected-error @below {{ambiguous "ClockDomain" association for port "o"}}
    out %o  : !firrtl.uint<1> domains []
  ) {
    // DI and DO are aliases
    firrtl.domain.define %DO, %DI : !firrtl.domain<@ClockDomain()>

    // o is on same domain as i
    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
  }
}

// Domain exported multiple times, this time with two outputs.
firrtl.circuit "DoubleExportOfDomain" {
  firrtl.domain @ClockDomain

  firrtl.extmodule @Generator(out D: !firrtl.domain<@ClockDomain()>)

  // expected-note @below {{in module "DoubleExportOfDomain"}}
  firrtl.module @DoubleExportOfDomain(
    // expected-note @below {{candidate association "D1"}}
    out %D1 : !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{candidate association "D2"}}
    out %D2 : !firrtl.domain<@ClockDomain()>,
    in  %i  : !firrtl.uint<1> domains [%D1],
    // expected-error @below {{ambiguous "ClockDomain" association for port "o"}}
    out %o  : !firrtl.uint<1> domains []
  ) {
    %gen_D = firrtl.instance gen @Generator(out D: !firrtl.domain<@ClockDomain()>)
    // DI and DO are aliases
    firrtl.domain.define %D1, %gen_D : !firrtl.domain<@ClockDomain()>
    firrtl.domain.define %D2, %gen_D : !firrtl.domain<@ClockDomain()>

    // o is on same domain as i
    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
  }
}

// InstanceChoice: Each module has different domains inferred.
// TODO: this just relies on the op-verifier for instance choice ops.
firrtl.circuit "ConflictingInstanceChoiceDomains" {
    firrtl.domain @ClockDomain

  firrtl.option @Option {
    firrtl.option_case @X
    firrtl.option_case @Y
  }

  // Foo's "out" port takes on the domains of "in1".
  firrtl.module @Foo(in %in1: !firrtl.uint<1>, in %in2: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    firrtl.connect %out, %in1 : !firrtl.uint<1>
  }

  // Bar's "out" port takes on the domains of "in2".
  // expected-note @below {{original module declared here}}
  firrtl.module @Bar(in %in1: !firrtl.uint<1>, in %in2: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    firrtl.connect %out, %in2 : !firrtl.uint<1>
  }

  firrtl.module @ConflictingInstanceChoiceDomains(in %in1: !firrtl.uint<1>, in %in2: !firrtl.uint<1>) {
    // expected-error @below {{'firrtl.instance_choice' op domain info for "out" must be '[2 : ui32]', but got '[0 : ui32]'}}
    %inst_in1, %inst_in2, %inst_out = firrtl.instance_choice inst @Foo alternatives @Option { @X -> @Foo, @Y -> @Bar } (in in1: !firrtl.uint<1>, in in2: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %inst_in1, %in1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %inst_in2, %in2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// Testing the reporting of multiple errors.
firrtl.circuit "MultipleErrors" {
  firrtl.domain @ClockDomain

  // We report the first error encountered in M1.
  firrtl.module @M1(
    // expected-note @below {{a has domains [A : ClockDomain]}}
    in  %a: !firrtl.uint<1> domains [%A],
    // expected-note @below {{b has domains [B : ClockDomain]}}
    out %b: !firrtl.uint<1> domains [%B],
    // expected-note @below {{input module port A declared here}}
    in  %A: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{input module port B declared here}}
    in  %B: !firrtl.domain<@ClockDomain()>
  ) {
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }

  // We report the first error encountered in M2.
  firrtl.module @M2(
    // expected-note @below {{a has domains [A : ClockDomain]}}
    in  %a: !firrtl.uint<1> domains [%A],
    // expected-note @below {{b has domains [B : ClockDomain]}}
    out %b: !firrtl.uint<1> domains [%B],
    // expected-note @below {{input module port A declared here}}
    in  %A: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{input module port B declared here}}
    in  %B: !firrtl.domain<@ClockDomain()>
  ) {
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }

  firrtl.module @MultipleErrors(
    in  %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%B],
    in  %A: !firrtl.domain<@ClockDomain()>,
    in  %B: !firrtl.domain<@ClockDomain()>
  ) {
    %inst1_a, %inst1_b, %inst1_A, %inst1_B = firrtl.instance inst1 @M1(
      in  a: !firrtl.uint<1> domains [A],
      out b: !firrtl.uint<1> domains [B],
      in  A: !firrtl.domain<@ClockDomain()>,
      in  B: !firrtl.domain<@ClockDomain()>
    ) 

    %inst2_a, %inst2_b, %inst2_A, %inst2_B = firrtl.instance inst2 @M2(
      in  a: !firrtl.uint<1> domains [A],
      out b: !firrtl.uint<1> domains [B],
      in  A: !firrtl.domain<@ClockDomain()>,
      in  B: !firrtl.domain<@ClockDomain()>
    ) 

    // unreported domain crossing here: we skip checking this module because
    // the child modules M1 and M2 have already reported errors.
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// Wire domain conflicts with connection (illegal domain crossing).
firrtl.circuit "WireDomainConflict" {
  firrtl.domain @ClockDomain
  firrtl.module @WireDomainConflict(
    // expected-note @below {{input module port A declared here}}
    in %A: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{input module port B declared here}}
    in %B: !firrtl.domain<@ClockDomain()>,
    // expected-note @below {{a has domains [A : ClockDomain]}}
    in %a: !firrtl.uint<1> domains [%A]
  ) {
    // expected-note @below {{w has domains [B : ClockDomain]}}
    %w = firrtl.wire domains[%B] : !firrtl.uint<1> domains[!firrtl.domain<@ClockDomain()>]
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.connect %w, %a : !firrtl.uint<1>
  }
}


// Trace domain sources through output ports.
firrtl.circuit "TraceThroughOutputPort" {
  firrtl.domain @ClockDomain

  firrtl.module @ClockGate(
    out %a: !firrtl.uint<1> domains [%A],
    // expected-note @below {{output module port A declared here}}
    out %A: !firrtl.domain<@ClockDomain()>
  ) {
    // expected-note @below {{%0 declared here}}
    %0 = firrtl.domain.create : !firrtl.domain<@ClockDomain()>
    // expected-note @below {{output module port A aliases %0}}
    firrtl.domain.define %A, %0 : !firrtl.domain<@ClockDomain()>
  }

  firrtl.module @TraceThroughOutputPort(
    // expected-note @below {{a has domains [A : ClockDomain]}}
    out %a: !firrtl.uint<1> domains [%A],
    // expected-note @below {{input module port A declared here}}
    in  %A: !firrtl.domain<@ClockDomain()>
  ) {
    // expected-note @below {{cg.a has domains [cg.A : ClockDomain]}}
    // expected-note @below {{output instance port cg.A declared here}}
    %cg_a, %cg_A = firrtl.instance cg @ClockGate(
      out a: !firrtl.uint<1> domains [A],
      out A: !firrtl.domain<@ClockDomain()>
    )
    // expected-error @below {{illegal domain crossing in operation}}
    firrtl.matchingconnect %a, %cg_a : !firrtl.uint<1>
  }
}
