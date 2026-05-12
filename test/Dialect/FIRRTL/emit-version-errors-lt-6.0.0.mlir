// Tests for --firrtl-version: ops requiring >= 6.0.0 error when targeting 5.0.0.
// RUN: circt-translate --export-firrtl --firrtl-version=5.0.0 \
// RUN:   --verify-diagnostics --split-input-file %s

// fflush statement requires >= 6.0.0
firrtl.circuit "FFlush" {
  firrtl.module @FFlush(in %clk : !firrtl.clock, in %en : !firrtl.uint<1>) {
    // expected-error @below {{'firrtl.fflush' op fflush requires FIRRTL 6.0.0}}
    firrtl.fflush %clk, %en : !firrtl.clock, !firrtl.uint<1>
  }
}

// -----

// fprintf statement requires >= 6.0.0
firrtl.circuit "FPrintF" {
  firrtl.module @FPrintF(in %clk : !firrtl.clock, in %en : !firrtl.uint<1>) {
    // expected-error @below {{'firrtl.fprintf' op fprintf requires FIRRTL 6.0.0}}
    firrtl.fprintf %clk, %en, "out.txt", "hello" : !firrtl.clock, !firrtl.uint<1>
  }
}

// -----

// Bool constant expression requires >= 6.0.0.  The constant is consumed by
// propassign (valid at 5.0.0 >= 3.1.0); the error fires on the bool op.
firrtl.circuit "BoolConst" {
  firrtl.module @BoolConst(out %b : !firrtl.bool) {
    // expected-error @below {{'firrtl.bool' op Bools requires FIRRTL 6.0.0}}
    %0 = firrtl.bool true
    firrtl.propassign %b, %0 : !firrtl.bool
  }
}

// -----

// Double constant expression requires >= 6.0.0.
firrtl.circuit "DoubleConst" {
  firrtl.module @DoubleConst(out %d : !firrtl.double) {
    // expected-error @below {{'firrtl.double' op Doubles requires FIRRTL 6.0.0}}
    %0 = firrtl.double 3.14
    firrtl.propassign %d, %0 : !firrtl.double
  }
}

// -----

// Unresolved path expression requires >= 6.0.0.
firrtl.circuit "UnresolvedPath" {
  firrtl.module @UnresolvedPath(out %p : !firrtl.path) {
    // expected-error @below {{'firrtl.unresolved_path' op Paths requires FIRRTL 6.0.0}}
    %0 = firrtl.unresolved_path "OMDeleted:"
    firrtl.propassign %p, %0 : !firrtl.path
  }
}

// -----

// known layers on an extmodule require >= 6.0.0.  The layer declaration is
// emitted without error (layers are valid at 5.0.0 >= 3.3.0); the error fires
// on the extmodule.
firrtl.circuit "KnownLayers" {
  firrtl.layer @GroupA bind {}
  // expected-error @below {{'firrtl.extmodule' op known layers requires FIRRTL 6.0.0}}
  firrtl.extmodule @ExtWithKnownLayers() attributes {knownLayers = [@GroupA]}
  firrtl.module @KnownLayers() {}
}

// -----

// prop.eq expression requires >= 6.0.0.
firrtl.circuit "PropEq" {
  firrtl.module @PropEq(in %a : !firrtl.string, in %b : !firrtl.string,
                        out %eq : !firrtl.bool) {
    // expected-error @below {{'firrtl.prop.eq' op property equality requires FIRRTL 6.0.0}}
    %0 = firrtl.prop.eq %a, %b : !firrtl.string
    firrtl.propassign %eq, %0 : !firrtl.bool
  }
}

// -----

// bool.and expression requires >= 6.0.0.
firrtl.circuit "BoolAnd" {
  firrtl.module @BoolAnd(in %a : !firrtl.bool, in %b : !firrtl.bool,
                         out %out : !firrtl.bool) {
    // expected-error @below {{'firrtl.bool.and' op boolean and requires FIRRTL 6.0.0}}
    %0 = firrtl.bool.and %a, %b
    firrtl.propassign %out, %0 : !firrtl.bool
  }
}

// -----

// bool.or expression requires >= 6.0.0.
firrtl.circuit "BoolOr" {
  firrtl.module @BoolOr(in %a : !firrtl.bool, in %b : !firrtl.bool,
                        out %out : !firrtl.bool) {
    // expected-error @below {{'firrtl.bool.or' op boolean or requires FIRRTL 6.0.0}}
    %0 = firrtl.bool.or %a, %b
    firrtl.propassign %out, %0 : !firrtl.bool
  }
}

// -----

// bool.xor expression requires >= 6.0.0.
firrtl.circuit "BoolXor" {
  firrtl.module @BoolXor(in %a : !firrtl.bool, in %b : !firrtl.bool,
                         out %out : !firrtl.bool) {
    // expected-error @below {{'firrtl.bool.xor' op boolean xor requires FIRRTL 6.0.0}}
    %0 = firrtl.bool.xor %a, %b
    firrtl.propassign %out, %0 : !firrtl.bool
  }
}
