// RUN: circt-opt --firrtl-probes-to-signals --verify-diagnostics --split-input-file %s

// Sending probe out from under a when is not possible without encompassing ExpandWhens.
// Detect and diagnose, and in practice use ExpandWhens first to ensure success.
firrtl.circuit "RefProducer" {
  // expected-note @below {{destination here}}
  firrtl.module @RefProducer(in %a: !firrtl.uint<4>, in %en: !firrtl.uint<1>, in %clk: !firrtl.clock, out %thereg: !firrtl.probe<uint>) attributes {convention = #firrtl<convention scalarized>} {
    firrtl.when %en : !firrtl.uint<1> {
      %myreg = firrtl.reg interesting_name %clk : !firrtl.clock, !firrtl.uint
      firrtl.connect %myreg, %a : !firrtl.uint, !firrtl.uint<4>
      // expected-note @below {{source here}}
      %0 = firrtl.ref.send %myreg : !firrtl.uint
      // expected-error @below {{unable to convert to equivalent connect}}
      firrtl.ref.define %thereg, %0 : !firrtl.probe<uint>
    }
  }
}

// -----

// `force_initial` and `release_initial` are not synthesized.
firrtl.circuit "RejectForceInitial" {
  firrtl.module @RejectForceInitial(in %val : !firrtl.uint<2>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %c1_ui1 = firrtl.constant 1 : !firrtl.const.uint<1>
    // expected-error @below {{force_initial not supported}}
    firrtl.ref.force_initial %c1_ui1, %w_ref, %val : !firrtl.const.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>
  }
}

// -----

// `release_initial` likewise.
firrtl.circuit "RejectReleaseInitial" {
  firrtl.module @RejectReleaseInitial() {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %c1_ui1 = firrtl.constant 1 : !firrtl.const.uint<1>
    // expected-error @below {{release_initial not supported}}
    firrtl.ref.release_initial %c1_ui1, %w_ref : !firrtl.const.uint<1>, !firrtl.rwprobe<uint<2>>
  }
}

// -----

firrtl.circuit "ExtOpenAgg" {
  firrtl.extmodule @ExtOpenAgg(
      // expected-error @below {{open aggregates not supported, cannot convert type}}
      out out: !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>>)
}

// -----

firrtl.circuit "LayerProbe" {
  firrtl.layer @Layer bind {}
  // expected-error @below {{layer-colored probes not supported, cannot convert type}}
  firrtl.module @LayerProbe(out %p: !firrtl.probe<uint<1>, @Layer>) {
    firrtl.layerblock @Layer {
      %w = firrtl.wire : !firrtl.uint<1>
      %w_p = firrtl.ref.send %w : !firrtl.uint<1>
      %w_p_l = firrtl.ref.cast %w_p : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @Layer>
      firrtl.ref.define %p, %w_p_l : !firrtl.probe<uint<1>, @Layer>
    }
  }
}

// -----

// A force through a `ref.cast` that changes the probed type lands on the copy
// wire the cast lowers to, which cannot drive the real target.  Diagnose
// instead of silently dropping the force.
firrtl.circuit "ForceThroughWideningCast" {
  firrtl.module @ForceThroughWideningCast(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    // expected-note @below {{target is reached through this op}}
    %cast = firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<8>>) -> !firrtl.rwprobe<uint>
    // expected-error @below {{unsupported force/release: cannot route force control to the target through this probe}}
    firrtl.ref.force %clock, %en, %cast, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint>, !firrtl.uint
  }
}

// -----

// The state machine for a probe exported from an instance lives in the child,
// and its control bundle carries one `forcedValue` for the whole target, so a
// force of a single field (through `ref.sub`) cannot be routed to it.
firrtl.circuit "ForceSubOfInstanceProbe" {
  firrtl.module @SubChild(out %p: !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>) {
    // expected-error @below {{force/release of aggregate types is not supported; compile with preserve-aggregates=none}}
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
  }
  firrtl.module @ForceSubOfInstanceProbe(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    %c_p = firrtl.instance c @SubChild(out p: !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>)
    // expected-note @below {{target is reached through this op}}
    %sub = firrtl.ref.sub %c_p[0] : !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    // expected-error @below {{unsupported force/release: cannot route force control to the target through this probe}}
    firrtl.ref.force %clock, %en, %sub, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Exporting a forceable probe through a type-changing `ref.cast` is legal when
// the circuit has no force/release: the port maps to a plain data type and the
// cast becomes a connect.  Forcing through such a cast is still diagnosed
// (see ForceThroughWideningCast above).
firrtl.circuit "ExportThroughWideningCast" {
  firrtl.module @ExportThroughWideningCast(out %p: !firrtl.rwprobe<uint>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %cast = firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<8>>) -> !firrtl.rwprobe<uint>
    firrtl.ref.define %p, %cast : !firrtl.rwprobe<uint>
  }
}

// -----

// Same restriction for a release (not just a force) of a field of an instance's
// probe.
firrtl.circuit "ReleaseSubOfInstanceProbe" {
  firrtl.module @RelSubChild(out %p: !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>) {
    // expected-error @below {{force/release of aggregate types is not supported; compile with preserve-aggregates=none}}
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
  }
  firrtl.module @ReleaseSubOfInstanceProbe(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    %c_p = firrtl.instance c @RelSubChild(out p: !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>)
    // expected-note @below {{target is reached through this op}}
    %sub = firrtl.ref.sub %c_p[1] : !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    // expected-error @below {{unsupported force/release: cannot route force control to the target through this probe}}
    firrtl.ref.release %clock, %en, %sub : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
  }
}

// -----

// An extmodule has no body to hold a state machine and its ports cannot be
// rewritten to carry force control, so a force of one of its forceable probes
// could never reach the target.  Diagnose instead of dropping it silently.
firrtl.circuit "ForceExtmoduleProbe" {
  firrtl.extmodule @Ext(out p: !firrtl.rwprobe<uint<8>>)
  firrtl.module @ForceExtmoduleProbe(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // expected-note @below {{target is a probe of this instance, whose module has no body to carry the force control}}
    %e_p = firrtl.instance e @Ext(out p: !firrtl.rwprobe<uint<8>>)
    // expected-error @below {{unsupported force/release: cannot route force control to the target through this probe}}
    firrtl.ref.force %clock, %en, %e_p, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// ... and for a release.
firrtl.circuit "ReleaseExtmoduleProbe" {
  firrtl.extmodule @Ext(out p: !firrtl.rwprobe<uint<8>>)
  firrtl.module @ReleaseExtmoduleProbe(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // expected-note @below {{target is a probe of this instance, whose module has no body to carry the force control}}
    %e_p = firrtl.instance e @Ext(out p: !firrtl.rwprobe<uint<8>>)
    // expected-error @below {{unsupported force/release: cannot route force control to the target through this probe}}
    firrtl.ref.release %clock, %en, %e_p : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
  }
}

// -----

// A field select of an extmodule's forceable probe inherits the instance as the
// blocker rather than reporting the `ref.sub`.
firrtl.circuit "ForceSubOfExtmoduleProbe" {
  firrtl.extmodule @Ext(out p: !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>)
  firrtl.module @ForceSubOfExtmoduleProbe(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // expected-note @below {{target is a probe of this instance, whose module has no body to carry the force control}}
    %e_p = firrtl.instance e @Ext(out p: !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>)
    %sub = firrtl.ref.sub %e_p[0] : !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    // expected-error @below {{unsupported force/release: cannot route force control to the target through this probe}}
    firrtl.ref.force %clock, %en, %sub, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Re-exporting an extmodule's forceable probe is legal when the circuit has no
// force/release: both ports map to a plain data type and the define becomes a
// connect.  Forcing an extmodule probe is still diagnosed (see
// ForceExtmoduleProbe above).
firrtl.circuit "ExportExtmoduleProbe" {
  firrtl.extmodule @Ext(out p: !firrtl.rwprobe<uint<8>>)
  firrtl.module @ExportExtmoduleProbe(out %p: !firrtl.rwprobe<uint<8>>) {
    %e_p = firrtl.instance e @Ext(out p: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.define %p, %e_p : !firrtl.rwprobe<uint<8>>
  }
}

// -----

// An `instance_choice` is body-less if *any* of its callees is: the force could
// land on the extmodule alternative.
firrtl.circuit "ForceChoiceWithExtmodule" {
  firrtl.option @Platform { firrtl.option_case @A }
  firrtl.module @ChoiceImpl(out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
  }
  firrtl.extmodule @ChoiceExtImpl(out p: !firrtl.rwprobe<uint<8>>)
  firrtl.module @ForceChoiceWithExtmodule(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // expected-note @below {{target is a probe of this instance, whose module has no body to carry the force control}}
    %e_p = firrtl.instance_choice e @ChoiceImpl alternatives @Platform { @A -> @ChoiceExtImpl } (out p: !firrtl.rwprobe<uint<8>>)
    // expected-error @below {{unsupported force/release: cannot route force control to the target through this probe}}
    firrtl.ref.force %clock, %en, %e_p, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Instance choices of forceable probes are legal when the circuit has no
// force/release: the port count is unchanged, so the instance_choice signature
// does not need extra control ports.  Forcing through an
// instance_choice that includes an extmodule is still diagnosed (see
// ForceChoiceWithExtmodule above).
firrtl.circuit "ForceChoiceUnsupported" {
  firrtl.option @Platform { firrtl.option_case @A }
  firrtl.module @ChoiceImpl(out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
  }
  firrtl.extmodule @ChoiceExtImpl(out p: !firrtl.rwprobe<uint<8>>)
  firrtl.module @ForceChoiceUnsupported(out %o: !firrtl.uint<8>) {
    %e_p = firrtl.instance_choice e @ChoiceImpl alternatives @Platform { @A -> @ChoiceExtImpl } (out p: !firrtl.rwprobe<uint<8>>)
    %r = firrtl.ref.resolve %e_p : !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %o, %r : !firrtl.uint<8>
  }
}

// -----

// Forcing a FIELD of a local aggregate (`ref.sub` of a local target) is
// diagnosed just like a field of an instance's probe: the state machine's
// control bundle carries a `forcedValue` sized for the whole target, so a
// force of a single field cannot be routed to it.
firrtl.circuit "ForceFieldOfLocalAggregate" {
  firrtl.module @ForceFieldOfLocalAggregate(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    // expected-note @below {{target is reached through this op}}
    %sub = firrtl.ref.sub %w_ref[0] : !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    // expected-error @below {{unsupported force/release: cannot route force control to the target through this probe}}
    firrtl.ref.force %clock, %en, %sub, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Forcing a whole aggregate is diagnosed: only ground-type targets are
// supported, so an aggregate must be lowered (e.g. `preserve-aggregates=none`)
// before this pass runs.
firrtl.circuit "ForceWholeLocalAggregate" {
  firrtl.module @ForceWholeLocalAggregate(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.bundle<a: uint<8>, b: uint<8>>, out %oa: !firrtl.uint<8>) {
    // expected-error @below {{force/release of aggregate types is not supported; compile with preserve-aggregates=none}}
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    %wa = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<8>, b: uint<8>>
    firrtl.matchingconnect %oa, %wa : !firrtl.uint<8>
    firrtl.ref.force %clock, %en, %w_ref, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>, !firrtl.bundle<a: uint<8>, b: uint<8>>
  }
}

// -----

// The clocked layer of one target is reduced into state elements running on a
// single clock, so two clocked accesses on *different* clocks cannot be
// synthesized: they used to be silently merged onto the first clock.
firrtl.circuit "MixedClocksOneTarget" {
  firrtl.module @MixedClocksOneTarget(in %clockA: !firrtl.clock, in %clockB: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>, out %o: !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>
    // expected-note @below {{target is also accessed on a different clock here}}
    firrtl.ref.force %clockA, %en, %w_ref, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // expected-error @below {{unsupported force/release: target is accessed on more than one clock}}
    firrtl.ref.release %clockB, %en, %w_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
  }
}
