// Tests for --firrtl-version: ops requiring >= 5.1.0 error when targeting 5.0.0.
// RUN: circt-translate --export-firrtl --firrtl-version=5.0.0 \
// RUN:   --verify-diagnostics --split-input-file %s

// fflush statement requires >= 5.1.0
firrtl.circuit "FFlush" {
  firrtl.module @FFlush(in %clk : !firrtl.clock, in %en : !firrtl.uint<1>) {
    // expected-error @below {{'firrtl.fflush' op fflush requires FIRRTL 5.1.0}}
    firrtl.fflush %clk, %en : !firrtl.clock, !firrtl.uint<1>
  }
}

// -----

// fprintf statement requires >= 5.1.0
firrtl.circuit "FPrintF" {
  firrtl.module @FPrintF(in %clk : !firrtl.clock, in %en : !firrtl.uint<1>) {
    // expected-error @below {{'firrtl.fprintf' op fprintf requires FIRRTL 5.1.0}}
    firrtl.fprintf %clk, %en, "out.txt", "hello" : !firrtl.clock, !firrtl.uint<1>
  }
}

// -----

// Bool constant expression requires >= 5.1.0.  The constant is consumed by
// propassign (valid at 5.0.0 >= 3.1.0); the error fires on the bool op.
firrtl.circuit "BoolConst" {
  firrtl.module @BoolConst(out %b : !firrtl.bool) {
    // expected-error @below {{'firrtl.bool' op Bools requires FIRRTL 5.1.0}}
    %0 = firrtl.bool true
    firrtl.propassign %b, %0 : !firrtl.bool
  }
}

// -----

// Double constant expression requires >= 5.1.0.
firrtl.circuit "DoubleConst" {
  firrtl.module @DoubleConst(out %d : !firrtl.double) {
    // expected-error @below {{'firrtl.double' op Doubles requires FIRRTL 5.1.0}}
    %0 = firrtl.double 3.14
    firrtl.propassign %d, %0 : !firrtl.double
  }
}

// -----

// Unresolved path expression requires >= 5.1.0.
firrtl.circuit "UnresolvedPath" {
  firrtl.module @UnresolvedPath(out %p : !firrtl.path) {
    // expected-error @below {{'firrtl.unresolved_path' op Paths requires FIRRTL 5.1.0}}
    %0 = firrtl.unresolved_path "OMDeleted:"
    firrtl.propassign %p, %0 : !firrtl.path
  }
}

// -----

// Unknown property expression requires >= 5.1.0.
firrtl.circuit "UnknownValue" {
  firrtl.module @UnknownValue(out %v : !firrtl.integer) {
    // expected-error @below {{'firrtl.unknown' op unknown property expressions requires FIRRTL 5.1.0}}
    %0 = firrtl.unknown : !firrtl.integer
    firrtl.propassign %v, %0 : !firrtl.integer
  }
}

// -----

// unsafe_domain_cast expression requires >= 5.1.0.  Consumed by firrtl.node
// (not version-gated); the error fires on the cast op.
firrtl.circuit "UnsafeDomainCast" {
  firrtl.module @UnsafeDomainCast(in %a : !firrtl.uint<1>) {
    // expected-error @below {{'firrtl.unsafe_domain_cast' op unsafe_domain_cast requires FIRRTL 5.1.0}}
    %0 = firrtl.unsafe_domain_cast %a : !firrtl.uint<1>
    %n = firrtl.node %0 : !firrtl.uint<1>
  }
}

// -----

// option group declaration requires >= 5.1.0
firrtl.circuit "OptionDecl" {
  // expected-error @below {{'firrtl.option' op option groups requires FIRRTL 5.1.0}}
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }
  firrtl.module @OptionDecl() {}
}

// -----

// domain declaration requires >= 5.1.0
firrtl.circuit "DomainDecl" {
  // expected-error @below {{'firrtl.domain' op domains requires FIRRTL 5.1.0}}
  firrtl.domain @ClockDomain
  firrtl.module @DomainDecl() {}
}

// -----

// simulation test declaration requires >= 5.1.0.  The circuit must contain a
// module matching its name (@SimulationTest) for FIRRTL verification; the
// simulation op and the referenced extmodule are placed so the emitter
// encounters the simulation op first.
firrtl.circuit "SimulationTest" {
  // expected-error @below {{'firrtl.simulation' op simulation tests requires FIRRTL 5.1.0}}
  firrtl.simulation @myTest, @SimTop {}
  firrtl.module @SimulationTest() {}
  firrtl.extmodule @SimTop(
    in clock: !firrtl.clock,
    in init: !firrtl.uint<1>,
    out done: !firrtl.uint<1>,
    out success: !firrtl.uint<1>
  )
}

// -----

// known layers on an extmodule require >= 5.1.0.  The layer declaration is
// emitted without error (layers are valid at 5.0.0 >= 3.3.0); the error fires
// on the extmodule.
firrtl.circuit "KnownLayers" {
  firrtl.layer @GroupA bind {}
  // expected-error @below {{'firrtl.extmodule' op known layers requires FIRRTL 5.1.0}}
  firrtl.extmodule @ExtWithKnownLayers() attributes {knownLayers = [@GroupA]}
  firrtl.module @KnownLayers() {}
}
