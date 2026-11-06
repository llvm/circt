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

// -----

// A field force through a probe wire cannot use whole-target force control.
firrtl.circuit "ForceFieldThroughProbeWire" {
  firrtl.module @ForceFieldThroughProbeWire(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    // expected-note @below {{target is reached through this op}}
    %sub = firrtl.ref.sub %w_ref[0] : !firrtl.rwprobe<bundle<a: uint<8>, b: uint<8>>>
    %pw = firrtl.wire : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %pw, %sub : !firrtl.rwprobe<uint<8>>
    // expected-error @below {{unsupported force/release: cannot route force control to the target through this probe}}
    firrtl.ref.force %clock, %en, %pw, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// A cycle of probe wires has no target.
firrtl.circuit "ForceProbeWireCycle" {
  firrtl.module @ForceProbeWireCycle(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    %a = firrtl.wire : !firrtl.rwprobe<uint<8>>
    %b = firrtl.wire : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %a, %b : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %b, %a : !firrtl.rwprobe<uint<8>>
    // expected-error @below {{unsupported force/release: unable to determine the target}}
    firrtl.ref.force %clock, %en, %a, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// One target may not be forced or released from two clock domains, even
// through aliasing probes.
firrtl.circuit "ForceTwoClockDomains" {
  firrtl.module @ForceTwoClockDomains(in %c1: !firrtl.clock, in %c2: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>, out %o: !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %p = firrtl.wire : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %w, %v : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>
    // expected-note @below {{other access here}}
    firrtl.ref.force %c1, %en, %w_ref, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // expected-error @below {{target is forced or released from different clock domains}}
    firrtl.ref.release %c2, %en, %p : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
  }
}

// -----

// Aliased ports stay aliased when re-exported through a module.
firrtl.circuit "ForceForwardedPortsTwoClockDomains" {
  firrtl.module private @Child(in %v: !firrtl.uint<8>, out %o: !firrtl.uint<8>, out %p1: !firrtl.rwprobe<uint<8>>, out %p2: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %w, %v : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>
    firrtl.ref.define %p1, %w_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %p2, %w_ref : !firrtl.rwprobe<uint<8>>
  }
  firrtl.module private @Mid(in %v: !firrtl.uint<8>, out %o: !firrtl.uint<8>, out %q1: !firrtl.rwprobe<uint<8>>, out %q2: !firrtl.rwprobe<uint<8>>) {
    %c_v, %c_o, %c_p1, %c_p2 = firrtl.instance c @Child(in v: !firrtl.uint<8>, out o: !firrtl.uint<8>, out p1: !firrtl.rwprobe<uint<8>>, out p2: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %c_v, %v : !firrtl.uint<8>
    firrtl.matchingconnect %o, %c_o : !firrtl.uint<8>
    firrtl.ref.define %q1, %c_p1 : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %q2, %c_p2 : !firrtl.rwprobe<uint<8>>
  }
  firrtl.module @ForceForwardedPortsTwoClockDomains(in %c1: !firrtl.clock, in %c2: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>, out %o: !firrtl.uint<8>) {
    %m_v, %m_o, %m_q1, %m_q2 = firrtl.instance m @Mid(in v: !firrtl.uint<8>, out o: !firrtl.uint<8>, out q1: !firrtl.rwprobe<uint<8>>, out q2: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %m_v, %v : !firrtl.uint<8>
    firrtl.matchingconnect %o, %m_o : !firrtl.uint<8>
    // expected-note @below {{other access here}}
    firrtl.ref.force %c1, %en, %m_q1, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // expected-error @below {{target is forced or released from different clock domains}}
    firrtl.ref.release %c2, %en, %m_q2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
  }
}

// -----

// A local access and an access through a port are checked against each other.
firrtl.circuit "ForceLocalAndPortTwoClockDomains" {
  firrtl.module private @Child(in %ck: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>, out %o: !firrtl.uint<8>, out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %w, %v : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
    // expected-note @below {{other access here}}
    firrtl.ref.release %ck, %en, %w_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
  }
  firrtl.module @ForceLocalAndPortTwoClockDomains(in %c1: !firrtl.clock, in %c2: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>, out %o: !firrtl.uint<8>) {
    %c_ck, %c_en, %c_v, %c_o, %c_p = firrtl.instance c @Child(in ck: !firrtl.clock, in en: !firrtl.uint<1>, in v: !firrtl.uint<8>, out o: !firrtl.uint<8>, out p: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %c_ck, %c1 : !firrtl.clock
    firrtl.matchingconnect %c_en, %en : !firrtl.uint<1>
    firrtl.matchingconnect %c_v, %v : !firrtl.uint<8>
    firrtl.matchingconnect %o, %c_o : !firrtl.uint<8>
    // expected-error @below {{target is forced or released from different clock domains}}
    firrtl.ref.force %c2, %en, %c_p, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// Accesses through an extmodule probe are checked for one clock domain.
firrtl.circuit "ExtForceTwoClocks" {
  firrtl.extmodule @ExtRW(out rw: !firrtl.rwprobe<uint<2>>)
  firrtl.module @ExtForceTwoClocks(in %c1: !firrtl.clock, in %c2: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<2>) {
    %e_rw = firrtl.instance e @ExtRW(out rw: !firrtl.rwprobe<uint<2>>)
    // expected-note @below {{other access here}}
    firrtl.ref.force %c1, %en, %e_rw, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>, !firrtl.uint<2>
    // expected-error @below {{target is forced or released from different clock domains}}
    firrtl.ref.release %c2, %en, %e_rw : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<2>>
  }
}

// -----

// Forcing a clock would mux the clock path, so a force and a release of a
// clock are each rejected, even when the clock is gated.
firrtl.circuit "ForceGatedClock" {
  firrtl.module @ForceGatedClock(in %clk: !firrtl.clock, in %gateEn: !firrtl.uint<1>, in %en: !firrtl.uint<1>, in %rel: !firrtl.uint<1>, in %d: !firrtl.uint<8>, out %o: !firrtl.uint<8>) {
    %cw = firrtl.wire sym @cw : !firrtl.clock
    %g = firrtl.int.clock_gate %clk, %gateEn
    firrtl.matchingconnect %cw, %g : !firrtl.clock
    %r = firrtl.reg %cw : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.matchingconnect %o, %r : !firrtl.uint<8>
    %cw_ref = firrtl.ref.rwprobe <@ForceGatedClock::@cw> : !firrtl.rwprobe<clock>
    %zero = firrtl.specialconstant 0 : !firrtl.clock
    // expected-error @below {{unsupported force/release of a clock: the synthesized override would mux the clock}}
    firrtl.ref.force %clk, %en, %cw_ref, %zero : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<clock>, !firrtl.clock
    // expected-error @below {{unsupported force/release of a clock: the synthesized override would mux the clock}}
    firrtl.ref.release %clk, %rel, %cw_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<clock>
  }
}

// -----

// A public forceable probe port of a clock could be forced from outside the
// circuit, so it is rejected; a private one is forced only from inside.
firrtl.circuit "PublicClockProbePort" {
  firrtl.module private @Child(in %clk: !firrtl.clock sym @clk, out %p: !firrtl.rwprobe<clock>) {
    %0 = firrtl.ref.rwprobe <@Child::@clk> : !firrtl.rwprobe<clock>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<clock>
  }
  // expected-error @below {{unsupported forceable probe of a clock on public module port "clockProbe": it could be forced from outside the circuit, which would mux the clock}}
  firrtl.module @PublicClockProbePort(in %clock: !firrtl.clock, out %clockProbe: !firrtl.rwprobe<clock>) {
    %c_clk, %c_p = firrtl.instance c @Child(in clk: !firrtl.clock, out p: !firrtl.rwprobe<clock>)
    firrtl.matchingconnect %c_clk, %clock : !firrtl.clock
    firrtl.ref.define %clockProbe, %c_p : !firrtl.rwprobe<clock>
  }
}
