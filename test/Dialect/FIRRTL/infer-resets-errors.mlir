// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-resets))' --verify-diagnostics --split-input-file %s

// Tests extracted from:
// - github.com/chipsalliance/firrtl:
//   - test/scala/firrtlTests/InferResetsSpec.scala
// - github.com/sifive/$internal:
//   - test/scala/firrtl/FullAsyncResetTransform.scala

//===----------------------------------------------------------------------===//
// Reset Inference
//===----------------------------------------------------------------------===//


// Should NOT allow last connect semantics to pick the right type for Reset
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "reset0" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %reset0: !firrtl.asyncreset, in %reset1: !firrtl.uint<1>, out %out: !firrtl.reset) {
    %w0 = firrtl.wire : !firrtl.reset
    %w1 = firrtl.wire : !firrtl.reset
    firrtl.connect %w0, %reset0 : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %w1, %reset1 : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %out, %w0 : !firrtl.reset, !firrtl.reset
    firrtl.connect %out, %w1 : !firrtl.reset, !firrtl.reset
  }
}

// -----
// Should NOT support last connect semantics across whens
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "reset2" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %reset0: !firrtl.asyncreset, in %reset1: !firrtl.asyncreset, in %reset2: !firrtl.uint<1>, in %en: !firrtl.uint<1>, out %out: !firrtl.reset) {
    %w0 = firrtl.wire : !firrtl.reset
    %w1 = firrtl.wire : !firrtl.reset
    %w2 = firrtl.wire : !firrtl.reset
    firrtl.connect %w0, %reset0 : !firrtl.reset, !firrtl.asyncreset
    firrtl.connect %w1, %reset1 : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %w2, %reset2 : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %out, %w2 : !firrtl.reset, !firrtl.reset
    firrtl.when %en : !firrtl.uint<1>  {
      firrtl.connect %out, %w0 : !firrtl.reset, !firrtl.reset
    } else  {
      firrtl.connect %out, %w1 : !firrtl.reset, !firrtl.reset
    }
  }
}

// -----
// Should not allow different Reset Types to drive a single Reset
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "reset0" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %reset0: !firrtl.asyncreset, in %reset1: !firrtl.uint<1>, in %en: !firrtl.uint<1>, out %out: !firrtl.reset) {
    %w1 = firrtl.wire : !firrtl.reset
    %w2 = firrtl.wire : !firrtl.reset
    firrtl.connect %w1, %reset0 : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %w2, %reset1 : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %out, %w1 : !firrtl.reset, !firrtl.reset
    firrtl.when %en : !firrtl.uint<1>  {
      firrtl.connect %out, %w2 : !firrtl.reset, !firrtl.reset
    }
  }
}

// -----
// Should error if a ResetType driving UInt<1> infers to AsyncReset
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "in" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %in: !firrtl.asyncreset, out %out: !firrtl.uint<1>) {
    %w = firrtl.wire  : !firrtl.reset
    firrtl.connect %w, %in : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %out, %w : !firrtl.uint<1>, !firrtl.reset
  }
}

// -----
// Should error if a ResetType driving AsyncReset infers to UInt<1>
firrtl.circuit "top"   {
  // expected-error @+2 {{reset network "in" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %in: !firrtl.uint<1>, out %out: !firrtl.asyncreset) {
    %w = firrtl.wire  : !firrtl.reset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %w, %in : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %out, %w : !firrtl.asyncreset, !firrtl.reset
  }
}

// -----
// Should not allow ResetType as an Input
firrtl.circuit "top" {
  // expected-error @+2 {{reset network never driven with concrete type}}
  // expected-note @+1 {{here: }}
  firrtl.module @top(in %in: !firrtl.bundle<foo: reset>, out %out: !firrtl.reset) {
    // expected-note @+1 {{here: }}
    %0 = firrtl.subfield %in[foo] : !firrtl.bundle<foo: reset>
    firrtl.connect %out, %0 : !firrtl.reset, !firrtl.reset
  }
}

// -----
// Should not allow ResetType as an ExtModule output
firrtl.circuit "top" {
  firrtl.extmodule @ext(out out: !firrtl.bundle<foo: reset>)
  // expected-note @+1 {{here: }}
  firrtl.module @top(out %out: !firrtl.reset) {
    // expected-error @+2 {{reset network never driven with concrete type}}
    // expected-note @+1 {{here: }}
    %e_out = firrtl.instance e @ext(out out: !firrtl.bundle<foo: reset>)
    // expected-note @+1 {{here: }}
    %0 = firrtl.subfield %e_out[foo] : !firrtl.bundle<foo: reset>
    firrtl.connect %out, %0 : !firrtl.reset, !firrtl.reset
  }
}

// -----
// Should not allow Vecs to infer different Reset Types
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "out[]" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %reset0: !firrtl.asyncreset, in %reset1: !firrtl.uint<1>, out %out: !firrtl.vector<reset, 2>) {
    %0 = firrtl.subindex %out[0] : !firrtl.vector<reset, 2>
    %1 = firrtl.subindex %out[1] : !firrtl.vector<reset, 2>
    firrtl.connect %0, %reset0 : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %1, %reset1 : !firrtl.reset, !firrtl.uint<1>
  }
}

// -----
// Should not allow an invalidated Wire to drive both a UInt<1> and an AsyncReset
firrtl.circuit "top" {
  // expected-error @+2 {{reset network "in0" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @top(in %in0: !firrtl.asyncreset, in %in1: !firrtl.uint<1>, out %out0: !firrtl.reset, out %out1: !firrtl.reset) {
    %w = firrtl.wire  : !firrtl.reset
    %invalid_reset = firrtl.invalidvalue : !firrtl.reset
    firrtl.connect %w, %invalid_reset : !firrtl.reset, !firrtl.reset
    firrtl.connect %out0, %w : !firrtl.reset, !firrtl.reset
    firrtl.connect %out1, %w : !firrtl.reset, !firrtl.reset
    firrtl.connect %out0, %in0 : !firrtl.reset, !firrtl.asyncreset
    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %out1, %in1 : !firrtl.reset, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "UninferredReset" {
  // expected-error @+2 {{a port "reset" with abstract reset type was unable to be inferred by InferResets}}
  // expected-note @+1 {{the module with this uninferred reset port was defined here}}
  firrtl.module @UninferredReset(in %reset: !firrtl.reset) {}
}

// -----

firrtl.circuit "UninferredRefReset" {
  firrtl.module @UninferredRefReset() {}
  // expected-error @+2 {{a port "reset" with abstract reset type was unable to be inferred by InferResets}}
  // expected-note @+1 {{the module with this uninferred reset port was defined here}}
  firrtl.module private @UninferredRefResetPriv(out %reset: !firrtl.probe<reset>) {}
}

// -----
// Should error when instance_choice has conflicting reset types between alternatives
firrtl.circuit "InstanceChoiceConflict" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }

  // Default target uses async reset
  firrtl.module @DefaultTarget(in %reset: !firrtl.reset) {
    %localReset = firrtl.wire : !firrtl.reset
    firrtl.matchingconnect %localReset, %reset : !firrtl.reset
  }

  // FPGA target uses sync reset
  firrtl.module @FPGATarget(in %reset: !firrtl.reset) {
    %localReset = firrtl.wire : !firrtl.reset
    firrtl.matchingconnect %localReset, %reset : !firrtl.reset
  }

  // expected-error @+2 {{reset network "asyncReset" simultaneously connected to async and sync resets}}
  // expected-note @+1 {{majority of connections to this reset are async}}
  firrtl.module @InstanceChoiceConflict(in %asyncReset: !firrtl.asyncreset, in %syncReset: !firrtl.uint<1>) {
    %w0 = firrtl.wire : !firrtl.reset
    %w1 = firrtl.wire : !firrtl.reset

    // Connect async reset to w0
    firrtl.connect %w0, %asyncReset : !firrtl.reset, !firrtl.asyncreset

    // expected-note @+1 {{sync drive here:}}
    firrtl.connect %w1, %syncReset : !firrtl.reset, !firrtl.uint<1>

    // Instance choice with conflicting reset connections
    %inst_reset = firrtl.instance_choice inst @DefaultTarget alternatives @Platform {
      @FPGA -> @FPGATarget
    } (in reset: !firrtl.reset)

    // Connect both async and sync resets through the instance
    firrtl.matchingconnect %inst_reset, %w0 : !firrtl.reset
    firrtl.matchingconnect %inst_reset, %w1 : !firrtl.reset
  }
}
