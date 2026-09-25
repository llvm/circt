// RUN: circt-opt --firrtl-probes-to-signals --cse --split-input-file %s | FileCheck %s

// Covers local and hierarchical force/release lowering, control-port routing,
// priority, aliases, casts, gated clocks, and unsupported targets.


// -----
// TEST 1: Force + release of a register in the same module.

// CHECK-LABEL: firrtl.module @SameModuleRegisterForceRelease
firrtl.circuit "SameModuleRegisterForceRelease" {
  firrtl.module @SameModuleRegisterForceRelease(in %clock: !firrtl.clock,in %enable: !firrtl.uint<1>, in %release: !firrtl.uint<1>, in %value: !firrtl.uint<8>, in %reset: !firrtl.uint<1>, out %o: !firrtl.uint<8>) {
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    %next = firrtl.constant 1 : !firrtl.uint<8>
    // CHECK: %r = firrtl.regreset %clock, %reset, %c0_ui8
    // CHECK: %r_forced = firrtl.wire : !firrtl.uint<8>
    %r, %r_ref = firrtl.regreset %clock, %reset, %c0 forceable : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %next : !firrtl.uint<8>
    firrtl.matchingconnect %o, %r : !firrtl.uint<8>

    // Force, then release.
    firrtl.ref.force %clock, %enable, %r_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %release, %r_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // The force must not become part of the register's next-state driver.
    // CHECK: firrtl.matchingconnect %r, %c1_ui8
    // Readers observe the override instead.
    // CHECK: firrtl.matchingconnect %o, %r_forced
    // The release is later, so it has priority: the force predicate is gated by
    // !release and releaseActive is the release predicate itself.
    // CHECK-DAG: %[[NR:.+]] = firrtl.not %release
    // CHECK-DAG: firrtl.and %enable, %[[NR]]
    // A single force needs only the `forced` flag; its RHS remains live.
    // CHECK: %{{.+}} = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<8>
    // The override uses the force's live RHS.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %value, %r)
    // CHECK: firrtl.matchingconnect %r_forced, %[[OVR]]
  }
}

// -----
// TEST 2: Multiple force/release to the same wire (priority order).

// CHECK-LABEL: firrtl.module @MultipleForceReleaseSameWire
firrtl.circuit "MultipleForceReleaseSameWire" {
  firrtl.module @MultipleForceReleaseSameWire(in %clock: !firrtl.clock, in %en1: !firrtl.uint<1>, in %en2: !firrtl.uint<1>, in %val1: !firrtl.uint<8>, in %val2: !firrtl.uint<8>, out %o: !firrtl.uint<8>) {
    // CHECK: %w = firrtl.wire : !firrtl.uint<8>
    // CHECK: %w_forced = firrtl.wire : !firrtl.uint<8>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>

    // Earlier accesses have lower priority.
    firrtl.ref.force %clock, %en1, %w_ref, %val1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.force %clock, %en2, %w_ref, %val2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %en2, %w_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // The wire keeps its own single driver.
    // CHECK: firrtl.matchingconnect %w, %c0_ui8
    // CHECK: firrtl.matchingconnect %o, %w_forced
    // The later release masks both forces; the later force masks the first.
    // CHECK: %[[NR:.+]] = firrtl.not %en2
    // CHECK: %[[F2:.+]] = firrtl.and %en2, %[[NR]]
    // CHECK: %[[F1:.+]] = firrtl.and %en1, %[[NR]]
    // CHECK: %[[FA:.+]] = firrtl.or %[[F1]], %[[F2]]
    // Latch the winning force while keeping both RHS values live.
    // CHECK: %forceWinner = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[FV:.+]] = firrtl.mux(%forceWinner, %val2, %val1)
    // CHECK: %[[WIN:.+]] = firrtl.mux(%[[FA]], %[[F2]], %forceWinner)
    // CHECK: firrtl.matchingconnect %forceWinner, %[[WIN]]
    // State registers have a power-on value of 0.
    // CHECK: %{{.+}} = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %{{.+}}_{{[0-9]+}} = firrtl.reg
    // Unforced reads fall back to the wire.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[FV]], %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 3: Different RWProbe values for one target share one state machine.

// CHECK-LABEL: firrtl.module @ForceReleaseSplitRWProbes
firrtl.circuit "ForceReleaseSplitRWProbes" {
  firrtl.module @ForceReleaseSplitRWProbes(
      in %clock:      !firrtl.clock,
      in %reset:      !firrtl.uint<1>,
      in %force_en:   !firrtl.uint<1>,
      in %release_en: !firrtl.uint<1>,
      in %val:        !firrtl.uint<8>,
      out %o:         !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire sym @w_sym forceable :
        !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>

    // A second RWProbe value for the same inner symbol.
    %w_ref2 = firrtl.ref.rwprobe <@ForceReleaseSplitRWProbes::@w_sym> :
        !firrtl.rwprobe<uint<8>>

    // Force and release use different RWProbe values for the same target.
    firrtl.ref.force %clock, %force_en, %w_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %release_en, %w_ref2 :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // One observed wire and one forced register must be emitted.
    // CHECK:     %w_forced = firrtl.wire : !firrtl.uint<8>
    // CHECK-NOT: %w_forced_{{[0-9]+}} = firrtl.wire
    // CHECK:     firrtl.matchingconnect %w, %c0_ui8
    // CHECK:     firrtl.matchingconnect %o, %w_forced

    // The release masks the force.
    // CHECK-DAG: %[[NR:.+]] = firrtl.not %release_en
    // CHECK-DAG: firrtl.and %force_en, %[[NR]]

    // CHECK:     %{{.+}} = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %{{.+}}_{{[0-9]+}} = firrtl.reg

    // One read-side override is emitted.
    // CHECK:     %[[OVR:.+]] = firrtl.mux(%forced, %val, %w)
    // CHECK:     firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 3b: Two independently-created rwprobes of the same inner symbol reuse
// one materialized hardware target.  The cache must be keyed by the target
// InnerRefAttr, not by either probe's SSA result.

// CHECK-LABEL: firrtl.module @RepeatedRWProbeTarget
firrtl.circuit "RepeatedRWProbeTarget" {
  firrtl.module @RepeatedRWProbeTarget(
      in %clock: !firrtl.clock,
      in %en: !firrtl.uint<1>,
      in %value: !firrtl.uint<8>,
      out %o: !firrtl.uint<8>) {
    %w = firrtl.wire sym @w_sym : !firrtl.uint<8>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>

    %ref1 = firrtl.ref.rwprobe <@RepeatedRWProbeTarget::@w_sym> :
        !firrtl.rwprobe<uint<8>>
    %ref2 = firrtl.ref.rwprobe <@RepeatedRWProbeTarget::@w_sym> :
        !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %en, %ref1, %value :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %en, %ref2 :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // Both accesses must reduce into one target state machine.
    // CHECK: %w_forced = firrtl.wire : !firrtl.uint<8>
    // CHECK-NOT: %w_forced_{{[0-9]+}} = firrtl.wire
    // CHECK: %[[OVR:.+]] = firrtl.mux(%{{.+}}, %value, %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 4: Different RWProbe values for one regreset share one state machine.

// CHECK-LABEL: firrtl.module @ForceReleaseSplitRWProbesReg
firrtl.circuit "ForceReleaseSplitRWProbesReg" {
  firrtl.module @ForceReleaseSplitRWProbesReg(
      in %clock:      !firrtl.clock,
      in %reset:      !firrtl.uint<1>,
      in %next:       !firrtl.uint<8>,
      in %force_en:   !firrtl.uint<1>,
      in %release_en: !firrtl.uint<1>,
      in %val:        !firrtl.uint<8>,
      out %o:         !firrtl.uint<8>) {
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    %r, %r_ref = firrtl.regreset sym @r_sym %clock, %reset, %c0 forceable :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>,
        !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %next : !firrtl.uint<8>
    firrtl.matchingconnect %o, %r : !firrtl.uint<8>

    // A second RWProbe value for the same inner symbol.
    %r_ref2 = firrtl.ref.rwprobe <@ForceReleaseSplitRWProbesReg::@r_sym> :
        !firrtl.rwprobe<uint<8>>

    firrtl.ref.force %clock, %force_en, %r_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %release_en, %r_ref2 :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // CHECK:     %r_forced = firrtl.wire : !firrtl.uint<8>
    // CHECK-NOT: %r_forced_{{[0-9]+}} = firrtl.wire
    // CHECK:     firrtl.matchingconnect %r, %next
    // CHECK:     firrtl.matchingconnect %o, %r_forced
    // CHECK-DAG: %[[NR:.+]] = firrtl.not %release_en
    // CHECK-DAG: firrtl.and %force_en, %[[NR]]
    // CHECK:     %{{.+}} = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %{{.+}}_{{[0-9]+}} = firrtl.reg
    // The override reads the raw register.
    // CHECK:     %[[OVR:.+]] = firrtl.mux(%forced, %val, %r)
    // CHECK:     firrtl.matchingconnect %r_forced, %[[OVR]]
  }
}

// -----
// TEST 5: Force and release sharing one RWProbe produce one state machine.

// CHECK-LABEL: firrtl.module @ForceReleaseSameRWProbe
firrtl.circuit "ForceReleaseSameRWProbe" {
  firrtl.module @ForceReleaseSameRWProbe(
      in %clock:      !firrtl.clock,
      in %force_en:   !firrtl.uint<1>,
      in %release_en: !firrtl.uint<1>,
      in %val:        !firrtl.uint<8>,
      out %o:         !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable :
        !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>

    // Both operations use the same RWProbe.
    firrtl.ref.force %clock, %force_en, %w_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %release_en, %w_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // CHECK-DAG: %[[NR:.+]] = firrtl.not %release_en
    // CHECK-DAG: firrtl.and %force_en, %[[NR]]
    // CHECK-DAG: firrtl.matchingconnect %w, %c0_ui8
    // The state register is emitted after the target and control reduction.
    // CHECK:     %{{.+}} = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %{{.+}}_{{[0-9]+}} = firrtl.reg
    // CHECK:     firrtl.matchingconnect %w_forced, %{{.+}}
  }
}

// -----

// TEST 6: Exported RWProbe data and control ports stay in lockstep across
// multiple instances.

// CHECK-LABEL: firrtl.module @Child
// CHECK-SAME: out %probe_out: !firrtl.uint<8>, in %probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>

firrtl.circuit "MultiInst" {
  firrtl.module @Child(out %probe_out: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %probe_out, %w_ref : !firrtl.rwprobe<uint<8>>
  }

  // CHECK-LABEL: firrtl.module @MultiInst
  firrtl.module @MultiInst(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // Both instances carry data and control ports.
    // CHECK: firrtl.instance a @Child(out probe_out: !firrtl.uint<8>, in probe_out_force_ctrl: !firrtl.bundle<{{.*}}>)
    // CHECK: firrtl.instance b @Child(out probe_out: !firrtl.uint<8>, in probe_out_force_ctrl: !firrtl.bundle<{{.*}}>)
    // CHECK: firrtl.matchingconnect %{{.+}}, %en : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %{{.+}}, %en : !firrtl.uint<1>
    %a_probe = firrtl.instance a @Child(out probe_out: !firrtl.rwprobe<uint<8>>)
    %b_probe = firrtl.instance b @Child(out probe_out: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.force %clock, %en, %a_probe, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.force %clock, %en, %b_probe, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}


// -----
// TEST 7: The `forced` state is initialized to zero; the wide force value stays
// live.
//
// CHECK-LABEL: firrtl.module @ForceResetValueIsZero
firrtl.circuit "ForceResetValueIsZero" {
  firrtl.module @ForceResetValueIsZero(
      in %clock:  !firrtl.clock,
      in %reset:  !firrtl.uint<1>,
      in %enable: !firrtl.uint<1>,
      in %val:    !firrtl.uint<16>,
      out %o:     !firrtl.uint<16>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<16>, !firrtl.rwprobe<uint<16>>
    %c0 = firrtl.constant 0 : !firrtl.uint<16>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<16>
    firrtl.matchingconnect %o, %w : !firrtl.uint<16>

    firrtl.ref.force %clock, %enable, %w_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<16>>, !firrtl.uint<16>

    // The state register has a typed-zero initial value.
    // CHECK: %forced = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // Nothing of the probed type is registered.
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<16>
  }
}


// -----
// TEST 8: Local and hierarchical forces merge through three levels.

// CHECK-LABEL: firrtl.circuit "Middle"
firrtl.circuit "Middle" {
  // The leaf exports data and force control.
// CHECK: firrtl.module @Leaf(out %reg_probe: !firrtl.uint<8>, in %clock: !firrtl.clock, in %data_in: !firrtl.uint<8>, in %enable: !firrtl.uint<1>, in %reg_probe_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>)
  firrtl.module @Leaf(out %reg_probe: !firrtl.rwprobe<uint<8>>, in %clock: !firrtl.clock, in %data_in: !firrtl.uint<8>, in %enable: !firrtl.uint<1>) {
    %reg, %reg_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %reg, %data_in : !firrtl.uint<8>
    firrtl.ref.define %reg_probe, %reg_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %enable, %reg_ref, %data_in : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // The leaf overrides reads of its local register.
    // CHECK: %reg_forced = firrtl.wire : !firrtl.uint<8>
    // CHECK: firrtl.matchingconnect %reg, %data_in
    // CHECK: firrtl.matchingconnect %reg_probe, %reg_forced
    // Local control has priority over inbound control.
    // CHECK: %[[CTRL_FORCE:.+]] = firrtl.subfield %reg_probe_force_ctrl[forceActive]
    // CHECK: %[[CTRL_RELEASE:.+]] = firrtl.subfield %reg_probe_force_ctrl[releaseActive]
    // CHECK: %[[CTRL_VALUE:.+]] = firrtl.subfield %reg_probe_force_ctrl[forcedValue]
    // CHECK: %[[ANY_FORCE:.+]] = firrtl.or %{{.+}}, %enable
    // CHECK: %forceWinner = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[MERGED:.+]] = firrtl.mux(%forceWinner, %data_in, %[[CTRL_VALUE]])
    // CHECK: %[[LOCAL_NEXT:.+]] = firrtl.mux(%[[ANY_FORCE]], %enable, %forceWinner)
    // CHECK: firrtl.matchingconnect %forceWinner, %[[LOCAL_NEXT]]
    // CHECK: %{{.+}} = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[MERGED]], %reg)
    // CHECK: firrtl.matchingconnect %reg_forced, %[[OVR]]
  }

  // Middle instantiates Leaf and can force the register.
  // CHECK-LABEL: firrtl.module @Middle
  // Middle forwards the same data and control port shape.
  // CHECK-SAME: out %reg_probe_out: !firrtl.uint<8>
  // CHECK: firrtl.instance leaf @Leaf(
  // CHECK-SAME: in reg_probe_force_ctrl: !firrtl.bundle
  // Middle's local force has priority over inbound control.
  // CHECK: firrtl.or %{{.+}}, %enable_middle
  firrtl.module @Middle(out %reg_probe_out: !firrtl.rwprobe<uint<8>>, in %clock: !firrtl.clock, in %data_in: !firrtl.uint<8>, in %enable_middle: !firrtl.uint<1>, in %value_middle: !firrtl.uint<8>) {
    %leaf_probe, %leaf_clock, %leaf_data, %leaf_enable = firrtl.instance leaf @Leaf(out reg_probe: !firrtl.rwprobe<uint<8>>, in clock: !firrtl.clock, in data_in: !firrtl.uint<8>, in enable: !firrtl.uint<1>)
    firrtl.matchingconnect %leaf_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %leaf_data, %data_in : !firrtl.uint<8>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.matchingconnect %leaf_enable, %c1 : !firrtl.uint<1>

    // // Force from middle level
    firrtl.ref.force %clock, %enable_middle, %leaf_probe, %value_middle : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // // Pass probe up to parent
    firrtl.ref.define %reg_probe_out, %leaf_probe : !firrtl.rwprobe<uint<8>>
  }

  // CHECK-LABEL: firrtl.module @ThreeLevelHierarchy
  firrtl.module @ThreeLevelHierarchy(in %clock: !firrtl.clock, in %data_in: !firrtl.uint<8>, in %enable_middle: !firrtl.uint<1>, in %value_middle: !firrtl.uint<8>, in %enable_top: !firrtl.uint<1>, in %value_top: !firrtl.uint<8>) {
    %middle_probe, %middle_clock, %middle_data, %middle_enable, %middle_value = firrtl.instance middle @Middle(out reg_probe_out: !firrtl.rwprobe<uint<8>>, in clock: !firrtl.clock, in data_in: !firrtl.uint<8>, in enable_middle: !firrtl.uint<1>, in value_middle: !firrtl.uint<8>)
    firrtl.matchingconnect %middle_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %middle_data, %data_in : !firrtl.uint<8>
    firrtl.matchingconnect %middle_enable, %enable_middle : !firrtl.uint<1>
    firrtl.matchingconnect %middle_value, %value_middle : !firrtl.uint<8>

    firrtl.ref.force %clock, %enable_top, %middle_probe, %value_top : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: firrtl.instance middle @Middle(
    // CHECK-SAME: out reg_probe_out: !firrtl.uint<8>
    // CHECK-SAME: in reg_probe_out_force_ctrl: !firrtl.bundle
    // CHECK: firrtl.matchingconnect %{{.+}}, %enable_top : !firrtl.uint<1>
  }
}


// -----
// TEST 9: Force and release on a plain register affect reads only.

// CHECK-LABEL: firrtl.circuit "PlainRegForceRelease"
firrtl.circuit "PlainRegForceRelease" {
  firrtl.module @PlainRegForceRelease(
      in %clock: !firrtl.clock,
      in %next: !firrtl.uint<8>,
      in %en_force: !firrtl.uint<1>,
      in %en_release: !firrtl.uint<1>,
      in %val: !firrtl.uint<8>,
      out %o: !firrtl.uint<8>) {
    %r, %r_ref = firrtl.reg %clock forceable :
        !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %next : !firrtl.uint<8>
    firrtl.matchingconnect %o, %r : !firrtl.uint<8>

    firrtl.ref.force %clock, %en_force, %r_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %en_release, %r_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // CHECK: %r = firrtl.reg %clock
    // CHECK: %r_forced = firrtl.wire : !firrtl.uint<8>
    // The register keeps its single next-state driver.
    // CHECK: firrtl.matchingconnect %r, %next
    // CHECK-NOT: firrtl.matchingconnect %r, %
    // CHECK: firrtl.matchingconnect %o, %r_forced
    // The release masks the force.
    // CHECK-DAG: %[[NR:.+]] = firrtl.not %en_release
    // CHECK-DAG: firrtl.and %en_force, %[[NR]]
    // Override reads the raw register.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %val, %r)
    // CHECK: firrtl.matchingconnect %r_forced, %[[OVR]]
  }
}

// -----
// TEST 10: Multiple releases mask a single force.

// CHECK-LABEL: firrtl.circuit "MultipleReleasesSingleForce"
firrtl.circuit "MultipleReleasesSingleForce" {
  firrtl.module @MultipleReleasesSingleForce(
      in %clock: !firrtl.clock,
      in %en_f: !firrtl.uint<1>,
      in %en_r1: !firrtl.uint<1>,
      in %en_r2: !firrtl.uint<1>,
      in %en_r3: !firrtl.uint<1>,
      in %val: !firrtl.uint<8>,
      out %o: !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>

    firrtl.ref.force %clock, %en_f, %w_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %en_r1, %w_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
    firrtl.ref.release %clock, %en_r2, %w_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
    firrtl.ref.release %clock, %en_r3, %w_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // The force is gated by the absence of every release.
    // CHECK-DAG: %[[ANY0:.+]] = firrtl.or %en_r3, %en_r2
    // CHECK-DAG: %[[ANY1:.+]] = firrtl.or %[[ANY0]], %en_r1
    // CHECK-DAG: %[[NR:.+]] = firrtl.not %[[ANY1]]
    // CHECK-DAG: firrtl.and %en_f, %[[NR]]
  }
}

// -----

// TEST 11: Only one of two instance probes is forced; the other is tied off.

// CHECK-LABEL: firrtl.circuit "SelectiveForce"
firrtl.circuit "SelectiveForce" {
  // CHECK: firrtl.module @Child(out %probe_out: !firrtl.uint<8>, in %probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>)
  firrtl.module @Child(out %probe_out: !firrtl.rwprobe<uint<8>>) {
    %target, %target_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c42 = firrtl.constant 42 : !firrtl.uint<8>
    firrtl.matchingconnect %target, %c42 : !firrtl.uint<8>
    firrtl.ref.define %probe_out, %target_ref : !firrtl.rwprobe<uint<8>>
  }

  // CHECK: firrtl.module @SelectiveForce
  firrtl.module @SelectiveForce(
      in %clock: !firrtl.clock,
      in %enable: !firrtl.uint<1>,
      in %force_value: !firrtl.uint<8>,
      out %read_value: !firrtl.uint<8>) {

    // CHECK-NEXT: %a_probe_out, %a_probe_out_force_ctrl = firrtl.instance a @Child(out probe_out: !firrtl.uint<8>, in probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>)
    %a_probe = firrtl.instance a @Child(out probe_out: !firrtl.rwprobe<uint<8>>)

    // CHECK-NEXT: %b_probe_out, %b_probe_out_force_ctrl = firrtl.instance b @Child(out probe_out: !firrtl.uint<8>, in probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>)
    %b_probe = firrtl.instance b @Child(out probe_out: !firrtl.rwprobe<uint<8>>)

    firrtl.ref.force %clock, %enable, %a_probe, %force_value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // CHECK: firrtl.matchingconnect %read_value, %b_probe_out
    %b_read = firrtl.ref.resolve %b_probe : !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %read_value, %b_read : !firrtl.uint<8>

    // The forced instance receives active control fields.
    // CHECK: %[[FALSE:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[A_FA:.+]] = firrtl.subfield %a_probe_out_force_ctrl[forceActive]
    // CHECK-NEXT: %[[A_RA:.+]] = firrtl.subfield %a_probe_out_force_ctrl[releaseActive]
    // CHECK-NEXT: %[[A_FV:.+]] = firrtl.subfield %a_probe_out_force_ctrl[forcedValue]
    // CHECK-NEXT: %[[A_CLK:.+]] = firrtl.subfield %a_probe_out_force_ctrl[clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[A_FA]], %enable
    // CHECK-NEXT: firrtl.matchingconnect %[[A_RA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[A_FV]], %force_value
    // CHECK-NEXT: firrtl.matchingconnect %[[A_CLK]], %clock

    // The unforced instance receives inactive control fields and a zero clock.
    // CHECK: %[[ZEROCLK:.+]] = firrtl.specialconstant 0 : !firrtl.clock
    // CHECK: %[[B_FA:.+]] = firrtl.subfield %b_probe_out_force_ctrl[forceActive]
    // CHECK-NEXT: %[[B_RA:.+]] = firrtl.subfield %b_probe_out_force_ctrl[releaseActive]
    // CHECK-NEXT: %[[B_FV:.+]] = firrtl.subfield %b_probe_out_force_ctrl[forcedValue]
    // CHECK-NEXT: %[[B_CLK:.+]] = firrtl.subfield %b_probe_out_force_ctrl[clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[B_FA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[B_RA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[B_FV]], %{{.+}}
    // CHECK-NEXT: firrtl.matchingconnect %[[B_CLK]], %[[ZEROCLK]]
  }
}

// -----
// TEST 12: A same-type cast forwards an instance force to the child.

// CHECK-LABEL: firrtl.module @CastChild
// CHECK-SAME: out %probe_out: !firrtl.uint<8>, in %probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>
firrtl.circuit "ForceThroughCast" {
  firrtl.module @CastChild(out %probe_out: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0_ui8 : !firrtl.uint<8>
    firrtl.ref.define %probe_out, %w_ref : !firrtl.rwprobe<uint<8>>
    // CHECK: %forced = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<8>
  }

  // CHECK-LABEL: firrtl.module @ForceThroughCast
  firrtl.module @ForceThroughCast(in %clock: !firrtl.clock, in %enable: !firrtl.uint<1>, in %value: !firrtl.uint<8>) {
    // CHECK: %c_probe_out, %c_probe_out_force_ctrl = firrtl.instance c @CastChild(out probe_out: !firrtl.uint<8>, in probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>)
    %c_probe = firrtl.instance c @CastChild(out probe_out: !firrtl.rwprobe<uint<8>>)
    %cast = firrtl.ref.cast %c_probe : (!firrtl.rwprobe<uint<8>>) -> !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %enable, %cast, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // The cast must not create a local copy or tie off the child's control.
    // CHECK-NOT: firrtl.wire : !firrtl.uint<8>
    // CHECK-NOT: firrtl.specialconstant
    // CHECK-NOT: %forced = firrtl.reg
    // The force drives the child's ctrl subfields instead.
    // CHECK: %[[C_FA:.+]] = firrtl.subfield %c_probe_out_force_ctrl[forceActive]
    // CHECK-NEXT: %[[C_RA:.+]] = firrtl.subfield %c_probe_out_force_ctrl[releaseActive]
    // CHECK-NEXT: %[[C_FV:.+]] = firrtl.subfield %c_probe_out_force_ctrl[forcedValue]
    // CHECK-NEXT: %[[C_CLK:.+]] = firrtl.subfield %c_probe_out_force_ctrl[clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[C_FA]], %enable
    // CHECK-NEXT: firrtl.matchingconnect %[[C_RA]], %[[FALSE:.+]]
    // CHECK-NEXT: firrtl.matchingconnect %[[C_FV]], {{%.+}}
    // CHECK-NEXT: firrtl.matchingconnect %[[C_CLK]], %clock
  }
}
// -----
// TEST 13: A same-type cast preserves a local force target.

// CHECK-LABEL: firrtl.module @LocalForceThroughCast
firrtl.circuit "LocalForceThroughCast" {
  firrtl.module @LocalForceThroughCast(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %rel: !firrtl.uint<1>, in %value: !firrtl.uint<8>, out %o: !firrtl.uint<8>) {
    // CHECK: %w = firrtl.wire : !firrtl.uint<8>
    // CHECK-NEXT: %w_forced = firrtl.wire : !firrtl.uint<8>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>

    %cast = firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<8>>) -> !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %en, %cast, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %rel, %cast : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // No copy wire or control bundle is needed for a local target.
    // CHECK-NOT: firrtl.wire
    // The release masks the force.
    // CHECK-DAG: %[[NR:.+]] = firrtl.not %rel
    // CHECK-DAG: firrtl.and %en, %[[NR]]
    // CHECK: %{{.+}} = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<8>
    // The override reads the real target and uses the live RHS.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %value, %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 14: A same-type cast preserves an exported target's force path.

// CHECK-LABEL: firrtl.module @ExportCastChild
// CHECK-SAME: out %probe_out: !firrtl.uint<8>, in %probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>
firrtl.circuit "ForceExportedThroughCast" {
  firrtl.module @ExportCastChild(out %probe_out: !firrtl.rwprobe<uint<8>>) {
    // CHECK: %w = firrtl.wire : !firrtl.uint<8>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    %cast = firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<8>>) -> !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %probe_out, %cast : !firrtl.rwprobe<uint<8>>

    // The state machine and override remain on %w.
    // CHECK: %w_forced = firrtl.wire : !firrtl.uint<8>
    // CHECK: firrtl.matchingconnect %w, %c0_ui8
    // CHECK: firrtl.matchingconnect %probe_out, %w_forced
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<8>
    // The state uses the control-input clock.
    // CHECK: %[[CTRL_FV:.+]] = firrtl.subfield %probe_out_force_ctrl[forcedValue]
    // CHECK: %[[CTRL_CLK:.+]] = firrtl.subfield %probe_out_force_ctrl[clk]
    // CHECK: %forced = firrtl.reg %[[CTRL_CLK]] {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[CTRL_FV]], %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }

  // CHECK-LABEL: firrtl.module @ForceExportedThroughCast
  firrtl.module @ForceExportedThroughCast(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // CHECK: %c_probe_out, %c_probe_out_force_ctrl = firrtl.instance c @ExportCastChild(out probe_out: !firrtl.uint<8>, in probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>)
    %p = firrtl.instance c @ExportCastChild(out probe_out: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.force %clock, %en, %p, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[CTRL_FA:.+]] = firrtl.subfield %c_probe_out_force_ctrl[forceActive]
    // CHECK: firrtl.matchingconnect %[[CTRL_FA]], %en : !firrtl.uint<1>
  }
}

// -----
// TEST 15: A pure re-export forwards inbound control without a local tie-off.

// CHECK-LABEL: firrtl.circuit "ReExportNoLocalForce"
firrtl.circuit "ReExportNoLocalForce" {
  firrtl.module @RELeaf(out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
  }

  // CHECK-LABEL: firrtl.module @REMid
  // CHECK-SAME: out %p: !firrtl.uint<8>, in %p_force_ctrl: !firrtl.bundle
  firrtl.module @REMid(out %p: !firrtl.rwprobe<uint<8>>) {
    // Forward data and control directly.
    // CHECK: %leaf_p, %leaf_p_force_ctrl = firrtl.instance leaf @RELeaf(out p: !firrtl.uint<8>, in p_force_ctrl: !firrtl.bundle
    %lp = firrtl.instance leaf @RELeaf(out p: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.define %p, %lp : !firrtl.rwprobe<uint<8>>
    // Forward the control fields without local force.
    // CHECK: %[[MID_FA:.+]] = firrtl.subfield %p_force_ctrl[forceActive]
    // CHECK: %[[LEAF_FA:.+]] = firrtl.subfield %leaf_p_force_ctrl[forceActive]
    // CHECK: firrtl.matchingconnect %[[LEAF_FA]], %[[MID_FA]]
  }

  // CHECK-LABEL: firrtl.module @ReExportNoLocalForce
  firrtl.module @ReExportNoLocalForce(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // CHECK: %mid_p, %mid_p_force_ctrl = firrtl.instance mid @REMid(out p: !firrtl.uint<8>, in p_force_ctrl: !firrtl.bundle
    %mp = firrtl.instance mid @REMid(out p: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.force %clock, %en, %mp, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[CTRL_FA:.+]] = firrtl.subfield %mid_p_force_ctrl[forceActive]
    // CHECK: firrtl.matchingconnect %[[CTRL_FA]], %en : !firrtl.uint<1>
  }
}

// -----
// TEST 16: Per-result tie-off for two forceable probes on one instance.

// CHECK-LABEL: firrtl.module @MixedProbesOneInstance
firrtl.circuit "MixedProbesOneInstance" {
  firrtl.module @TwoProbes(out %pa: !firrtl.rwprobe<uint<8>>, out %pb: !firrtl.rwprobe<uint<4>>) {
    %a, %a_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %b, %b_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.define %pa, %a_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %pb, %b_ref : !firrtl.rwprobe<uint<4>>
  }

  firrtl.module @MixedProbesOneInstance(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // CHECK: %c_pa, %c_pb, %c_pa_force_ctrl, %c_pb_force_ctrl = firrtl.instance c @TwoProbes
    %pa, %pb = firrtl.instance c @TwoProbes(out pa: !firrtl.rwprobe<uint<8>>, out pb: !firrtl.rwprobe<uint<4>>)
    firrtl.ref.force %clock, %en, %pa, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[PA_FA:.+]] = firrtl.subfield %c_pa_force_ctrl[forceActive]
    // CHECK: %[[PA_CLK:.+]] = firrtl.subfield %c_pa_force_ctrl[clk]

    // The forced probe receives the real clock.
    // CHECK: firrtl.matchingconnect %[[PA_FA]], %en : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %[[PA_CLK]], %clock : !firrtl.clock

    // The unforced probe receives a constant-zero clock.
    // CHECK: %[[ZEROCLK:.+]] = firrtl.specialconstant 0 : !firrtl.clock
    // CHECK: %[[PB_CLK:.+]] = firrtl.subfield %c_pb_force_ctrl[clk]
    // CHECK: firrtl.matchingconnect %[[PB_CLK]], %[[ZEROCLK]]
    // CHECK-NOT: firrtl.specialconstant
  }
}

// -----
// TEST 17: A release-only local target uses an invalid forced value and zero
// force activity.

// CHECK-LABEL: firrtl.module @ReleaseOnlyLocal
firrtl.circuit "ReleaseOnlyLocal" {
  firrtl.module @ReleaseOnlyLocal(in %clock: !firrtl.clock, in %rel: !firrtl.uint<1>, out %o: !firrtl.uint<8>) {
    // CHECK: %w = firrtl.wire : !firrtl.uint<8>
    // CHECK-NEXT: %w_forced = firrtl.wire : !firrtl.uint<8>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>

    firrtl.ref.release %clock, %rel, %w_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // The invalid forced value is unreachable while force activity is zero.
    // CHECK: %[[INV:.+]] = firrtl.invalidvalue : !firrtl.uint<8>
    // No winner state is needed without a force.
    // CHECK: %{{.+}} = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %forceWinner = firrtl.reg
    // CHECK: %[[FALSE:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[NEXT:.+]] = firrtl.mux(%rel, %[[FALSE]], %forced)
    // CHECK: firrtl.matchingconnect %forced, %[[NEXT]]
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[INV]], %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 18: A release-only instance probe forwards release control.

// CHECK-LABEL: firrtl.circuit "ReleaseOnlyInstance"
firrtl.circuit "ReleaseOnlyInstance" {
  // CHECK: firrtl.module @ROChild
  // CHECK-SAME: out %p: !firrtl.uint<8>, in %p_force_ctrl: !firrtl.bundle
  firrtl.module @ROChild(out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
  }

  // CHECK-LABEL: firrtl.module @ReleaseOnlyInstance
  firrtl.module @ReleaseOnlyInstance(in %clock: !firrtl.clock, in %rel: !firrtl.uint<1>) {
    // CHECK: %c_p, %c_p_force_ctrl = firrtl.instance c @ROChild
    %p = firrtl.instance c @ROChild(out p: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.release %clock, %rel, %p : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // The release drives the control clock instead of a tie-off clock.
    // CHECK-NOT: firrtl.specialconstant
    // CHECK: %[[FALSE:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[INVALID:.+]] = firrtl.invalidvalue : !firrtl.uint<8>
    // CHECK: %[[FA:.+]] = firrtl.subfield %c_p_force_ctrl[forceActive]
    // CHECK-NEXT: %[[RA:.+]] = firrtl.subfield %c_p_force_ctrl[releaseActive]
    // CHECK-NEXT: %[[FV:.+]] = firrtl.subfield %c_p_force_ctrl[forcedValue]
    // CHECK-NEXT: %[[CLK:.+]] = firrtl.subfield %c_p_force_ctrl[clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[FA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[RA]], %rel
    // CHECK-NEXT: firrtl.matchingconnect %[[FV]], %[[INVALID]]
    // CHECK-NEXT: firrtl.matchingconnect %[[CLK]], %clock
  }
}

// -----
// TEST 19: A local release merges with inbound control on an exported target.

// CHECK-LABEL: firrtl.circuit "ReleaseOnlyExported"
firrtl.circuit "ReleaseOnlyExported" {
  // CHECK: firrtl.module @ROExportChild
  // CHECK-SAME: out %p: !firrtl.uint<8>, in %p_force_ctrl: !firrtl.bundle
  firrtl.module @ROExportChild(in %clock: !firrtl.clock, in %rel: !firrtl.uint<1>, out %p: !firrtl.rwprobe<uint<8>>) {
    // Read the control input directly.
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.release %clock, %rel, %w_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // Local release has priority over the inbound event.
    // CHECK: %[[IB_FA:.+]] = firrtl.subfield %p_force_ctrl[forceActive]
    // CHECK: %[[IB_RA:.+]] = firrtl.subfield %p_force_ctrl[releaseActive]
    // CHECK: %[[IB_FV:.+]] = firrtl.subfield %p_force_ctrl[forcedValue]
    // CHECK: %[[OR:.+]] = firrtl.or %{{.+}}, %rel
  }

  // CHECK-LABEL: firrtl.module @ReleaseOnlyExported
  firrtl.module @ReleaseOnlyExported(in %clock: !firrtl.clock, in %rel: !firrtl.uint<1>, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    %c_clock, %c_rel, %c_p = firrtl.instance c @ROExportChild(in clock: !firrtl.clock, in rel: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_rel, %rel : !firrtl.uint<1>
    firrtl.ref.force %clock, %en, %c_p, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[CTRL_FA:.+]] = firrtl.subfield %c_p_force_ctrl[forceActive]
    // CHECK: firrtl.matchingconnect %[[FA:.+]], %en : !firrtl.uint<1>
  }
}

// -----
// TEST 20: A same-type cast preserves a release to an instance probe.

// CHECK-LABEL: firrtl.circuit "ReleaseOnlyThroughCast"
firrtl.circuit "ReleaseOnlyThroughCast" {
  firrtl.module @RCChild(out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
  }

  // CHECK-LABEL: firrtl.module @ReleaseOnlyThroughCast
  firrtl.module @ReleaseOnlyThroughCast(in %clock: !firrtl.clock, in %rel: !firrtl.uint<1>) {
    // CHECK: %c_p, %c_p_force_ctrl = firrtl.instance c @RCChild
    %p = firrtl.instance c @RCChild(out p: !firrtl.rwprobe<uint<8>>)
    %cast = firrtl.ref.cast %p : (!firrtl.rwprobe<uint<8>>) -> !firrtl.rwprobe<uint<8>>
    firrtl.ref.release %clock, %rel, %cast : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // The cast needs no copy wire or local state machine.
    // CHECK-NOT: %forced = firrtl.reg
    // CHECK-NOT: firrtl.specialconstant
    // CHECK: %[[FALSE:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[FA:.+]] = firrtl.subfield %c_p_force_ctrl[forceActive]
    // CHECK-NEXT: %[[RA:.+]] = firrtl.subfield %c_p_force_ctrl[releaseActive]
    // CHECK-NEXT: %[[FV:.+]] = firrtl.subfield %c_p_force_ctrl[forcedValue]
    // CHECK-NEXT: %[[CLK:.+]] = firrtl.subfield %c_p_force_ctrl[clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[FA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[RA]], %rel
    // CHECK-NEXT: firrtl.matchingconnect %[[FV]], %{{.+}}
    // CHECK-NEXT: firrtl.matchingconnect %[[CLK]], %clock
  }
}

// -----
// TEST 21: A self-referential register next state reads the overridden value.

// CHECK-LABEL: firrtl.circuit "SelfReferentialReg"
firrtl.circuit "SelfReferentialReg" {
  // CHECK-LABEL: firrtl.module @SelfReferentialReg
  firrtl.module @SelfReferentialReg(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>, out %o: !firrtl.uint<8>) {
    // CHECK: %r = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    // CHECK-NEXT: %r_forced = firrtl.wire : !firrtl.uint<8>
    %r, %r_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c1 = firrtl.constant 1 : !firrtl.uint<8>
    %sum = firrtl.add %r, %c1 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    %next = firrtl.tail %sum, 1 : (!firrtl.uint<9>) -> !firrtl.uint<8>
    firrtl.matchingconnect %r, %next : !firrtl.uint<8>
    firrtl.matchingconnect %o, %r : !firrtl.uint<8>
    firrtl.ref.force %clock, %en, %r_ref, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // The adder operand is the observed value, not the raw register.
    // CHECK: %[[SUM:.+]] = firrtl.add %r_forced, %c1_ui8
    // CHECK: %[[NEXT:.+]] = firrtl.tail %[[SUM]], 1
    // CHECK: firrtl.matchingconnect %r, %[[NEXT]]
    // CHECK: firrtl.matchingconnect %o, %r_forced
    // The override reads the raw register to avoid a combinational loop.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %v, %r)
    // CHECK: firrtl.matchingconnect %r_forced, %[[OVR]]
  }
}

// -----
// TEST 24: A forced target with no reads gets no override.

// CHECK-LABEL: firrtl.circuit "ForcedButNeverRead"
firrtl.circuit "ForcedButNeverRead" {
  // CHECK-LABEL: firrtl.module @ForcedButNeverRead
  firrtl.module @ForcedButNeverRead(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // CHECK: %w = firrtl.wire : !firrtl.uint<8>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    // CHECK: %[[C0:.+]] = firrtl.constant 0 : !firrtl.uint<8>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.force %clock, %en, %w_ref, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // Preserve the original driver without an observed wire or mux.
    // CHECK-NOT: firrtl.wire
    // CHECK-NOT: firrtl.reg
    // CHECK: firrtl.matchingconnect %w, %[[C0]]
    // CHECK-NOT: firrtl.mux({{.*}}, %w)
  }
}

// -----
// TEST 25: Three clocked forces latch the winning source while keeping RHS
// values live.

// CHECK-LABEL: firrtl.circuit "ThreeForcesStickyValue"
firrtl.circuit "ThreeForcesStickyValue" {
  // CHECK-LABEL: firrtl.module @ThreeForcesStickyValue
  firrtl.module @ThreeForcesStickyValue(in %clock: !firrtl.clock, in %en1: !firrtl.uint<1>,
                                        in %en2: !firrtl.uint<1>, in %en3: !firrtl.uint<1>,
                                        in %v1: !firrtl.uint<8>, in %v2: !firrtl.uint<8>,
                                        in %v3: !firrtl.uint<8>, in %d: !firrtl.uint<8>,
                                        out %o: !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %w, %d : !firrtl.uint<8>
    firrtl.matchingconnect %o, %w : !firrtl.uint<8>
    firrtl.ref.force %clock, %en1, %w_ref, %v1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.force %clock, %en2, %w_ref, %v2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.force %clock, %en3, %w_ref, %v3 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // The force value is not snapshotted.
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<8>
    // Later forces mask earlier ones before updating the winner state.
    // CHECK: %[[NOT_EN3:.+]] = firrtl.not %en3
    // CHECK-NEXT: %[[SEL2:.+]] = firrtl.and %en2, %[[NOT_EN3]]
    // CHECK-NEXT: %[[EN23:.+]] = firrtl.or %en3, %en2
    // CHECK-NEXT: %[[NOT_EN23:.+]] = firrtl.not %[[EN23]]
    // CHECK-NEXT: %[[SEL1:.+]] = firrtl.and %en1, %[[NOT_EN23]]
    // CHECK-NEXT: %[[FA0:.+]] = firrtl.or %[[SEL1]], %[[SEL2]]
    // CHECK-NEXT: %[[FA:.+]] = firrtl.or %[[FA0]], %en3
    // The first force is the mux default; later forces have winner state.
    // CHECK: %forceWinner = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NEXT: %[[V2:.+]] = firrtl.mux(%forceWinner, %v2, %v1)
    // CHECK-NEXT: %forceWinner_0 = firrtl.reg %clock {initial = 0 : ui1, name = "forceWinner"} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NEXT: %[[V3:.+]] = firrtl.mux(%forceWinner_0, %v3, %[[V2]])
    // CHECK-NEXT: %[[WIN2:.+]] = firrtl.mux(%[[FA]], %[[SEL2]], %forceWinner)
    // CHECK-NEXT: firrtl.matchingconnect %forceWinner, %[[WIN2]]
    // CHECK-NEXT: %[[WIN3:.+]] = firrtl.mux(%[[FA]], %en3, %forceWinner_0)
    // CHECK-NEXT: firrtl.matchingconnect %forceWinner_0, %[[WIN3]]
    // The override uses the selected live value.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[V3]], %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 26: A forceable register on a gated clock is converted to base-clock
// state and an observed-value hold mux.

// CHECK-LABEL: firrtl.circuit "ForceGatedClockRegister"
firrtl.circuit "ForceGatedClockRegister" {
  // CHECK-LABEL: firrtl.module @ForceGatedClockRegister
  firrtl.module @ForceGatedClockRegister(in %clock: !firrtl.clock, in %gateEn: !firrtl.uint<1>,
                                         in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>,
                                         in %v: !firrtl.uint<8>, out %o: !firrtl.uint<8>) {
    %gated = firrtl.int.clock_gate %clock, %gateEn
    // The clock gate is gone; the register is rebound to the base clock.
    // CHECK: %r = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    // CHECK-NOT: firrtl.int.clock_gate
    // CHECK: %r_forced = firrtl.wire : !firrtl.uint<8>
    %r, %r_ref = firrtl.reg %gated forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.matchingconnect %o, %r : !firrtl.uint<8>

    firrtl.ref.force %clock, %en, %r_ref, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // The hold mux reads the observed value, preserving the force while gated.
    // CHECK: %[[HOLD:.+]] = firrtl.mux(%gateEn, %d, %r_forced)
    // CHECK: firrtl.matchingconnect %r, %[[HOLD]]
    // CHECK: firrtl.matchingconnect %o, %r_forced
    // CHECK: %forced = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %v, %r)
    // CHECK: firrtl.matchingconnect %r_forced, %[[OVR]]
  }
}

// -----
// TEST 27: A parent forces a child register whose local clock is gated.

// CHECK-LABEL: firrtl.circuit "ForceChildGatedClockRegister"
firrtl.circuit "ForceChildGatedClockRegister" {
  // The child exports data and force control.
  // CHECK: firrtl.module @GatedChild(out %p: !firrtl.uint<8>, in %clock: !firrtl.clock, in %gateEn: !firrtl.uint<1>, in %d: !firrtl.uint<8>, in %p_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>)
  firrtl.module @GatedChild(out %p: !firrtl.rwprobe<uint<8>>, in %clock: !firrtl.clock,
                            in %gateEn: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %gated = firrtl.int.clock_gate %clock, %gateEn
    // The gate is converted to base-clock state and a hold mux.
    // CHECK: %r = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    // CHECK-NOT: firrtl.int.clock_gate
    // CHECK: %r_forced = firrtl.wire : !firrtl.uint<8>
    %r, %r_ref = firrtl.reg %gated forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.ref.define %p, %r_ref : !firrtl.rwprobe<uint<8>>

    // The hold mux and exported data use the observed value.
    // CHECK: %[[HOLD:.+]] = firrtl.mux(%gateEn, %d, %r_forced)
    // CHECK: firrtl.matchingconnect %r, %[[HOLD]]
    // CHECK: firrtl.matchingconnect %p, %r_forced
    // CHECK: %[[CTRL_CLK:.+]] = firrtl.subfield %p_force_ctrl[clk]
    // CHECK: %forced = firrtl.reg %[[CTRL_CLK]] {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @ForceChildGatedClockRegister
  firrtl.module @ForceChildGatedClockRegister(in %clock: !firrtl.clock, in %gateEn: !firrtl.uint<1>,
                                              in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // CHECK: %c_p, %c_clock, %c_gateEn, %c_d, %c_p_force_ctrl = firrtl.instance c @GatedChild(out p: !firrtl.uint<8>, in clock: !firrtl.clock, in gateEn: !firrtl.uint<1>, in d: !firrtl.uint<8>, in p_force_ctrl: !firrtl.bundle
    %c_p, %c_clock, %c_gateEn, %c_d = firrtl.instance c @GatedChild(out p: !firrtl.rwprobe<uint<8>>, in clock: !firrtl.clock, in gateEn: !firrtl.uint<1>, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_gateEn, %gateEn : !firrtl.uint<1>

    // The parent's clock drives the child's control clock.
    firrtl.ref.force %clock, %en, %c_p, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[CTRL_FA:.+]] = firrtl.subfield %c_p_force_ctrl[forceActive]
    // CHECK-NEXT: %[[CTRL_RA:.+]] = firrtl.subfield %c_p_force_ctrl[releaseActive]
    // CHECK-NEXT: %[[CTRL_FV:.+]] = firrtl.subfield %c_p_force_ctrl[forcedValue]
    // CHECK-NEXT: %[[CTRL_CLK:.+]] = firrtl.subfield %c_p_force_ctrl[clk]
    // CHECK: firrtl.matchingconnect %[[CTRL_FA]], %en
    // CHECK-NEXT: firrtl.matchingconnect %[[CTRL_RA]], %{{.+}}
    // CHECK-NEXT: firrtl.matchingconnect %[[CTRL_FV]], %v
    // CHECK-NEXT: firrtl.matchingconnect %[[CTRL_CLK]], %clock
  }
}

// -----
// TEST 28: Two exported control channels reach one target state.

firrtl.circuit "TwoPortsBothForced" {
  // CHECK-LABEL: firrtl.module @TwoPortChild
  firrtl.module @TwoPortChild(out %p0: !firrtl.rwprobe<uint<8>>,
                              out %p1: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire sym @w forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p0, %w_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %p1, %w_ref : !firrtl.rwprobe<uint<8>>

    // Both exported connections observe the override.
    // CHECK: %w_forced = firrtl.wire : !firrtl.uint<8>
    // CHECK-NOT: %w_forced_{{[0-9]+}} = firrtl.wire
    // CHECK: %[[P0FA:.+]] = firrtl.subfield %p0_force_ctrl[forceActive]
    // CHECK: %[[P0FV:.+]] = firrtl.subfield %p0_force_ctrl[forcedValue]
    // CHECK: %[[P1FA:.+]] = firrtl.subfield %p1_force_ctrl[forceActive]
    // CHECK: %[[P1FV:.+]] = firrtl.subfield %p1_force_ctrl[forcedValue]
    // The later port has priority over the earlier port.
    // CHECK: %[[FA:.+]] = firrtl.or %{{.+}}, %[[P1FA]]
    // CHECK: %forceWinner = firrtl.reg
    // CHECK: %[[VALUE:.+]] = firrtl.mux(%forceWinner, %[[P1FV]], %[[P0FV]])
    // CHECK: %forced = firrtl.reg
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[VALUE]], %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }

  // CHECK-LABEL: firrtl.module @TwoPortsBothForced
  firrtl.module @TwoPortsBothForced(in %clock: !firrtl.clock,
                                    in %en0: !firrtl.uint<1>,
                                    in %en1: !firrtl.uint<1>,
                                    in %v0: !firrtl.uint<8>,
                                    in %v1: !firrtl.uint<8>) {
    %p0, %p1 = firrtl.instance child @TwoPortChild(
        out p0: !firrtl.rwprobe<uint<8>>, out p1: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.force %clock, %en0, %p0, %v0 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.force %clock, %en1, %p1, %v1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: firrtl.matchingconnect %{{.+}}, %en0
    // CHECK: firrtl.matchingconnect %{{.+}}, %en1
  }
}

// -----
// TEST 29: An unforced second exported port is tied off independently.

firrtl.circuit "TwoPortsFirstForced" {
  // CHECK-LABEL: firrtl.module @TwoPortFirstChild
  firrtl.module @TwoPortFirstChild(out %p0: !firrtl.rwprobe<uint<8>>,
                                   out %p1: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire sym @w forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p0, %w_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %p1, %w_ref : !firrtl.rwprobe<uint<8>>
    // CHECK: %w_forced = firrtl.wire : !firrtl.uint<8>
    // CHECK: %forced = firrtl.reg
  }

  // CHECK-LABEL: firrtl.module @TwoPortsFirstForced
  firrtl.module @TwoPortsFirstForced(in %clock: !firrtl.clock,
                                     in %en: !firrtl.uint<1>,
                                     in %v: !firrtl.uint<8>) {
    %p0, %p1 = firrtl.instance child @TwoPortFirstChild(
        out p0: !firrtl.rwprobe<uint<8>>, out p1: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.force %clock, %en, %p0, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: firrtl.matchingconnect %{{.+}}, %en
    // CHECK: firrtl.specialconstant 0 : !firrtl.clock
  }
}

// -----
// TEST 30: Local control has priority over two exported controls.

firrtl.circuit "TwoPortsAndLocalForce" {
  // CHECK-LABEL: firrtl.module @TwoPortLocalChild
  firrtl.module @TwoPortLocalChild(out %p0: !firrtl.rwprobe<uint<8>>,
                                   out %p1: !firrtl.rwprobe<uint<8>>,
                                   in %local_en: !firrtl.uint<1>,
                                   in %local_value: !firrtl.uint<8>,
                                   in %clock: !firrtl.clock) {
    %w, %w_ref = firrtl.wire sym @w forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p0, %w_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %p1, %w_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %local_en, %w_ref, %local_value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %forceWinner = firrtl.reg
    // CHECK: %forceWinner_0 = firrtl.reg
    // CHECK: %forced = firrtl.reg
  }

  // CHECK-LABEL: firrtl.module @TwoPortsAndLocalForce
  firrtl.module @TwoPortsAndLocalForce(in %clock: !firrtl.clock,
                                       in %en0: !firrtl.uint<1>,
                                       in %en1: !firrtl.uint<1>,
                                       in %v0: !firrtl.uint<8>,
                                       in %v1: !firrtl.uint<8>,
                                       in %local_en: !firrtl.uint<1>,
                                       in %local_value: !firrtl.uint<8>) {
    %p0, %p1, %child_local_en, %child_local_value, %child_clock = firrtl.instance child @TwoPortLocalChild(
        out p0: !firrtl.rwprobe<uint<8>>, out p1: !firrtl.rwprobe<uint<8>>,
        in local_en: !firrtl.uint<1>, in local_value: !firrtl.uint<8>,
        in clock: !firrtl.clock)
    firrtl.matchingconnect %child_local_en, %local_en : !firrtl.uint<1>
    firrtl.matchingconnect %child_local_value, %local_value : !firrtl.uint<8>
    firrtl.matchingconnect %child_clock, %clock : !firrtl.clock
    firrtl.ref.force %clock, %en0, %p0, %v0 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.force %clock, %en1, %p1, %v1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: firrtl.matchingconnect %{{.+}}, %en0
    // CHECK: firrtl.matchingconnect %{{.+}}, %en1
  }
}

// -----
// TEST 31: Layer-local tie-offs do not leak constants into module-scope state.

// CHECK-LABEL: firrtl.module @InstanceControlInLayerblock
firrtl.circuit "InstanceControlInLayerblock" {
  firrtl.layer @A bind {}

  firrtl.module @Leaf(out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
  }

  firrtl.module @InstanceControlInLayerblock(in %clock: !firrtl.clock,
                                             in %en: !firrtl.uint<1>,
                                             in %rel: !firrtl.uint<1>,
                                             in %v: !firrtl.uint<8>,
                                             out %o: !firrtl.uint<8>) {
    %x, %x_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %x, %c0 : !firrtl.uint<8>
    firrtl.matchingconnect %o, %x : !firrtl.uint<8>

    // Tie off the unused child control inside the layerblock.
    // CHECK: firrtl.layerblock @A
    firrtl.layerblock @A {
      // CHECK: %[[TIEOFF:.+]] = firrtl.constant 0 : !firrtl.uint<1>
      %p = firrtl.instance leaf @Leaf(out p: !firrtl.rwprobe<uint<8>>)
      // CHECK: firrtl.matchingconnect %{{.+}}, %[[TIEOFF]]
      // CHECK: firrtl.matchingconnect %{{.+}}, %[[TIEOFF]]
    }

    firrtl.ref.force %clock, %en, %x_ref, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %rel, %x_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // Module-scope reduction uses dominating constants from its own region.
    // CHECK: %[[NREL:.+]] = firrtl.not %rel
    // CHECK: firrtl.and %en, %[[NREL]]
    // CHECK: %[[ZERO:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[ONE:.+]] = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: firrtl.mux(%rel, %[[ZERO]], %forced)
    // CHECK: firrtl.mux(%{{.+}}, %[[ONE]], %{{.+}})
  }
}
