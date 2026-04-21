// Tests for --firrtl-version: ops requiring >= 3.3.0 error when targeting 3.1.0.
// RUN: circt-translate --export-firrtl --firrtl-version=3.1.0 \
// RUN:   --verify-diagnostics --split-input-file %s

// layer declaration requires >= 3.3.0
firrtl.circuit "LayerDecl" {
  // expected-error @below {{'firrtl.layer' op layers requires FIRRTL 3.3.0}}
  firrtl.layer @A bind {}
  firrtl.module @LayerDecl() {}
}

// -----

// layerblock statement requires >= 3.3.0.  The module is placed before the
// layer declaration so the emitter encounters the layerblock first; the layer
// reference is still found by the MLIR symbol table for IR verification.
firrtl.circuit "LayerBlock" {
  firrtl.module @LayerBlock() {
    // expected-error @below {{'firrtl.layerblock' op layers requires FIRRTL 3.3.0}}
    firrtl.layerblock @A {}
  }
  firrtl.layer @A bind {}
}
