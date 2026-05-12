// Tests for --firrtl-version: ops requiring >= 7.0.0 error when targeting 6.0.0.
// RUN: circt-translate --export-firrtl --firrtl-version=6.0.0 \
// RUN:   --verify-diagnostics --split-input-file %s

// Unknown property expression requires >= 7.0.0.
firrtl.circuit "UnknownValue" {
  firrtl.module @UnknownValue(out %v : !firrtl.integer) {
    // expected-error @below {{'firrtl.unknown' op unknown property expressions requires FIRRTL 7.0.0}}
    %0 = firrtl.unknown : !firrtl.integer
    firrtl.propassign %v, %0 : !firrtl.integer
  }
}

// -----

// unsafe_domain_cast expression requires >= 7.0.0.  Consumed by firrtl.node
// (not version-gated); the error fires on the cast op.
firrtl.circuit "UnsafeDomainCast" {
  firrtl.module @UnsafeDomainCast(in %a : !firrtl.uint<1>) {
    // expected-error @below {{'firrtl.unsafe_domain_cast' op unsafe_domain_cast requires FIRRTL 7.0.0}}
    %0 = firrtl.unsafe_domain_cast %a : !firrtl.uint<1>
    %n = firrtl.node %0 : !firrtl.uint<1>
  }
}

// -----

// option group declaration requires >= 7.0.0
firrtl.circuit "OptionDecl" {
  // expected-error @below {{'firrtl.option' op option groups requires FIRRTL 7.0.0}}
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }
  firrtl.module @OptionDecl() {}
}

// -----

// domain declaration requires >= 7.0.0
firrtl.circuit "DomainDecl" {
  // expected-error @below {{'firrtl.domain' op domains requires FIRRTL 7.0.0}}
  firrtl.domain @ClockDomain
  firrtl.module @DomainDecl() {}
}

// -----

// simulation test declaration requires >= 7.0.0.  The circuit must contain a
// module matching its name (@SimulationTest) for FIRRTL verification; the
// simulation op and the referenced extmodule are placed so the emitter
// encounters the simulation op first.
firrtl.circuit "SimulationTest" {
  // expected-error @below {{'firrtl.simulation' op simulation tests requires FIRRTL 7.0.0}}
  firrtl.simulation @myTest, @SimTop {}
  firrtl.module @SimulationTest() {}
  firrtl.extmodule @SimTop(
    in clock: !firrtl.clock,
    in init: !firrtl.uint<1>,
    out done: !firrtl.uint<1>,
    out success: !firrtl.uint<1>
  )
}
