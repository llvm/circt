// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-inliner))' -allow-unregistered-dialect -verify-diagnostics --split-input-file %s

// Reject inlining into when (run ExpandWhens first).

firrtl.circuit "InlineIntoWhen" {
  firrtl.module private @Child () attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {}
  firrtl.module @InlineIntoWhen(in %cond : !firrtl.uint<1>) {
    // expected-note @below {{containing operation 'firrtl.when' not safe to inline into}}
    firrtl.when %cond : !firrtl.uint<1> {
      // expected-error @below {{cannot inline instance}}
      firrtl.instance c @Child()
    }
  }
}

// -----

// Reject flattening through when (run ExpandWhens first).

firrtl.circuit "FlattenThroughWhen" {
  firrtl.module private @GChild () {}
  firrtl.module private @Child (in %cond : !firrtl.uint<1>) {
    // expected-note @below {{containing operation 'firrtl.when' not safe to inline into}}
    firrtl.when %cond : !firrtl.uint<1> {
      // expected-error @below {{cannot inline instance}}
      firrtl.instance c @GChild()
    }
  }
  firrtl.module @FlattenThroughWhen(in %cond : !firrtl.uint<1>) attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    %c_cond = firrtl.instance c @Child(in cond : !firrtl.uint<1>)
    firrtl.matchingconnect %c_cond, %cond : !firrtl.uint<1>
  }
}

// -----

// Reject inlining into unrecognized operations.

firrtl.circuit "InlineIntoIfdef" {
  sv.macro.decl @A_0["A"]
  firrtl.module private @Child () attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {}
  firrtl.module @InlineIntoIfdef() {
    // expected-note @below {{containing operation 'sv.ifdef' not safe to inline into}}
    sv.ifdef @A_0 {
      // expected-error @below {{cannot inline instance}}
      firrtl.instance c @Child()
    }
  }
}

// -----

// Conservatively reject cloning operations with regions that we don't recognize.

firrtl.circuit "InlineIfdef" {
  sv.macro.decl @A_0["A"]
  firrtl.module private @Child () attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    // expected-error @below {{unsupported operation 'sv.ifdef' cannot be inlined}}
    sv.ifdef @A_0 { }
  }
  firrtl.module @InlineIfdef() {
    firrtl.instance c @Child()
  }
}

// -----

// Cannot inline layers into layers.
// Presently the issue is detected by the verifier.

firrtl.circuit "InlineLayerIntoLayer" {
  firrtl.layer @I  inline {
    firrtl.layer @J  inline {
    }
  }
  firrtl.module private @MatchAgain(in %i: !firrtl.uint<8>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    // expected-error @below {{op has an un-nested layer symbol, but does not have a 'firrtl.module' op as a parent}}
    firrtl.layerblock @I {
      firrtl.layerblock @I::@J {
        %n = firrtl.node interesting_name %i : !firrtl.uint<8>
      }
    }
  }
  firrtl.module @InlineLayerIntoLayer(in %i: !firrtl.uint<8>) attributes {convention = #firrtl<convention scalarized>} {
    // expected-note @below {{illegal parent op defined here}}
    firrtl.layerblock @I {
      %c_i = firrtl.instance c interesting_name @MatchAgain(in i: !firrtl.uint<8>)
      firrtl.matchingconnect %c_i, %i : !firrtl.uint<8>
    }
  }
}

// -----

// Flatten/inline annotations are only meaningful on regular modules.

firrtl.circuit "FlattenExtModule" {
  // expected-error @below {{inline/flatten annotations are only valid on a regular module}}
  firrtl.extmodule private @Ext() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]}
  firrtl.module @FlattenExtModule() {
    firrtl.instance e @Ext()
  }
}

// -----

firrtl.circuit "InlineExtModule" {
  // expected-error @below {{inline/flatten annotations are only valid on a regular module}}
  firrtl.extmodule private @Ext() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}
  firrtl.module @InlineExtModule() {
    firrtl.instance e @Ext()
  }
}

// -----

// Class-likes are FModuleLike but live in the property world; inline/flatten
// annotations on them are rejected like on any other non-regular module.
firrtl.circuit "InlineClass" {
  // expected-error @below {{inline/flatten annotations are only valid on a regular module}}
  firrtl.class private @C() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {}
  firrtl.module @InlineClass() {
    %o = firrtl.object @C()
  }
}

// -----

// An inner reference smuggled into an annotation payload names another
// module; the inliner cannot know its semantics (its target may itself be
// inlined away), so inlining an op carrying one is rejected rather than
// silently relocating the clone around it.
firrtl.circuit "ForeignInnerRef" {
  firrtl.module private @A() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    // expected-error @below {{unsupported inner reference #hw.innerNameRef<@ForeignInnerRef::@tw> found while inlining}}
    %w = firrtl.wire {annotations = [
        {class = "test", target = #hw.innerNameRef<@ForeignInnerRef::@tw>}]} : !firrtl.uint<1>
  }
  firrtl.module @ForeignInnerRef() {
    %tw = firrtl.wire sym @tw : !firrtl.uint<1>
    firrtl.instance a @A()
  }
}

