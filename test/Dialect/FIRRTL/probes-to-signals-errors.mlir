// RUN: circt-opt --firrtl-probes-to-signals --verify-diagnostics --split-input-file %s

// Probe exports from conditional regions are rejected.
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

// Force/release inside a layerblock is rejected.
firrtl.circuit "RejectLayerForce" {
  firrtl.layer @Layer bind {}
  firrtl.module @RejectLayerForce(in %clock: !firrtl.clock,
                                  in %enable: !firrtl.uint<1>,
                                  in %value: !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>,
                                           !firrtl.rwprobe<uint<8>>
    firrtl.layerblock @Layer {
      // expected-error @below {{force inside a layerblock is not supported}}
      firrtl.ref.force %clock, %enable, %w_ref, %value : !firrtl.clock,
          !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    }
  }
}

// -----

// Force/release inside conditional regions is rejected.
firrtl.circuit "RejectWhenForce" {
  firrtl.module @RejectWhenForce(in %clock: !firrtl.clock,
                                 in %enable: !firrtl.uint<1>,
                                 in %value: !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>,
                                           !firrtl.rwprobe<uint<8>>
    firrtl.when %enable : !firrtl.uint<1> {
      // expected-error @below {{force inside a when or match block is not supported}}
      firrtl.ref.force %clock, %enable, %w_ref, %value : !firrtl.clock,
          !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    }
  }
}

// -----

firrtl.circuit "RejectWhenRelease" {
  firrtl.module @RejectWhenRelease(in %clock: !firrtl.clock,
                                   in %enable: !firrtl.uint<1>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>,
                                           !firrtl.rwprobe<uint<8>>
    firrtl.when %enable : !firrtl.uint<1> {
      // expected-error @below {{release inside a when or match block is not supported}}
      firrtl.ref.release %clock, %enable, %w_ref : !firrtl.clock,
          !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
    }
  }
}

// -----

firrtl.circuit "RejectMatchForce" {
  firrtl.module @RejectMatchForce(in %clock: !firrtl.clock,
                                  in %enable: !firrtl.uint<1>,
                                  in %value: !firrtl.uint<8>,
                                  in %tag: !firrtl.enum<Only: uint<1>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>,
                                           !firrtl.rwprobe<uint<8>>
    firrtl.match %tag : !firrtl.enum<Only: uint<1>> {
      case Only(%caseTag) {
        // expected-error @below {{force inside a when or match block is not supported}}
        firrtl.ref.force %clock, %enable, %w_ref, %value : !firrtl.clock,
            !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
      }
    }
  }
}

// -----

// A type-changing cast cannot carry force control to the original target.
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

// A type-changing cast cannot carry an exported force-control port.
firrtl.circuit "ExportThroughWideningCast" {
  firrtl.module @ExportThroughWideningCast(out %p: !firrtl.rwprobe<uint>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    // expected-note @below {{target is reached through this op}}
    %cast = firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<8>>) -> !firrtl.rwprobe<uint>
    // expected-error @below {{forceable probe port cannot be lowered: force control cannot be routed to the target through this probe}}
    firrtl.ref.define %p, %cast : !firrtl.rwprobe<uint>
  }
}


// -----

// A field force cannot use whole-target force control.
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

// A field-level inner symbol cannot carry whole-target force control.
firrtl.circuit "ForceFieldOfRWProbe" {
  firrtl.module @ForceFieldOfRWProbe(in %clock: !firrtl.clock,
                                     in %en: !firrtl.uint<1>,
                                     in %v: !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire sym [<@field_sym, 1, public>] forceable :
        !firrtl.bundle<a: uint<8>, b: uint<8>>,
        !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    // expected-note @below {{target is reached through this op}}
    %r = firrtl.ref.rwprobe <@ForceFieldOfRWProbe::@field_sym> :
        !firrtl.rwprobe<uint<8>>
    // expected-error @below {{unsupported force/release: cannot route force control to the target through this probe}}
    firrtl.ref.force %clock, %en, %r, %v : !firrtl.clock, !firrtl.uint<1>,
        !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Whole-aggregate force is rejected; only ground targets are supported.
firrtl.circuit "ForceWholeLocalAggregate" {
  firrtl.module @ForceWholeLocalAggregate(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.bundle<a: uint<8>, b: uint<8>>, out %oa: !firrtl.uint<8>) {
    // expected-error @below {{force/release of aggregate types is not supported; compile with preserve-aggregate=none}}
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    %wa = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<8>, b: uint<8>>
    firrtl.matchingconnect %oa, %wa : !firrtl.uint<8>
    firrtl.ref.force %clock, %en, %w_ref, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>, !firrtl.bundle<a: uint<8>, b: uint<8>>
  }
}

// -----

// Module output ports cannot use read-side force overrides.
firrtl.circuit "ForceOutputPort" {
  firrtl.module @ForceOutputPort(in %clock: !firrtl.clock,
                                 in %en: !firrtl.uint<1>,
                                 in %value: !firrtl.uint<8>,
                                 // expected-error @below {{cannot synthesize force/release: target is a module output port}}
                                 out %o: !firrtl.uint<8> sym @osym) {
    %r = firrtl.ref.rwprobe <@ForceOutputPort::@osym> : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %en, %r, %value : !firrtl.clock, !firrtl.uint<1>,
        !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}
