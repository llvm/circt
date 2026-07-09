// RUN: circt-opt --firrtl-probes-to-signals --cse --split-input-file %s | FileCheck %s

// This test file covers force/release synthesis for the ProbesToSignals pass.
// The pass synthesizes force/release operations into a per-probe state machine:
// - a `forced` register (UInt<1>) tracking whether the target is forced,
// - a `forcedValue` register holding the value to drive when forced,
// - an override mux injected at the target's connect point, and
// - a control bundle wire {forceActive, releaseActive, forcedValue, clk} whose
//   fields are driven exactly once by the priority reduction.
//
// Test scenarios covered:
//  1. Force/release of a register in the same module.
//  2. Multiple force/release operations to the same target (priority ordering).
//  3. Mix of clocked force and force_initial (the initial access rides along on
//     the clocked access's clock).
//  4. Force/release targeting the same wire via DIFFERENT rwprobe SSA values
//     (post-ExpandWhens shape) collapse into one state machine.
//  5. Same split-rwprobe scenario on a register.
//  6. Force + release sharing the SAME rwprobe SSA value (non-bug path).
//  7. A probe exported by a module instantiated more than once (lockstep ports).
//  8. Reset values of the state-machine registers are typed zeros.
//  9. Three-level hierarchy: forces from leaf, middle, and top compose.
// 10. Force + release on a plain (no-reset) register.
// 11. Multiple releases reduced against a single force.
// 12. Un-forced RWProbe.

// -----
// TEST 1: Force + release of a register in the same module.

// CHECK-LABEL: firrtl.module @SameModuleRegisterForceRelease
firrtl.circuit "SameModuleRegisterForceRelease" {
  firrtl.module @SameModuleRegisterForceRelease(in %clock: !firrtl.clock,in %enable: !firrtl.uint<1>, in %release: !firrtl.uint<1>, in %value: !firrtl.uint<8>, in %reset: !firrtl.uint<1>) {
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    %next = firrtl.constant 1 : !firrtl.uint<8>
    // CHECK: %r = firrtl.regreset
    %r, %r_ref = firrtl.regreset %clock, %reset, %c0 forceable : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %next : !firrtl.uint<8>

    // Force, then release.
    firrtl.ref.force %clock, %enable, %r_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %release, %r_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // The module has a `reset` port, so the state-machine registers are created
    // as `regreset` with a synchronous reset to 0 (prevents X-initialization).
    // CHECK: %forced = firrtl.regreset {{.+}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: %forcedValue = firrtl.regreset {{.+}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    // forceActive = enable AND forceWins.
    // CHECK-DAG: firrtl.and %enable, %{{.+}}
    // releaseActive is gated by !forceWins so a concurrent force suppresses it.
    // CHECK-DAG: %[[NFW:.+]] = firrtl.not %{{.+}}
    // CHECK-DAG: firrtl.and %release, %[[NFW]]
    // The register's next-value driver is replaced by the override mux output.
    // CHECK: firrtl.mux(%forced, %forcedValue, %{{.+}})
    // CHECK: firrtl.matchingconnect %r, %{{.+}}
  }
}

// -----
// TEST 2: Multiple force/release to the same wire (priority order).

// CHECK-LABEL: firrtl.module @MultipleForceReleaseSameWire
firrtl.circuit "MultipleForceReleaseSameWire" {
  firrtl.module @MultipleForceReleaseSameWire(in %clock: !firrtl.clock, in %en1: !firrtl.uint<1>, in %en2: !firrtl.uint<1>, in %val1: !firrtl.uint<8>, in %val2: !firrtl.uint<8>) {
    // CHECK: %w = firrtl.wire : !firrtl.uint<8>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>

    // First force (lowest priority - earliest in PriorityMux chain).
    firrtl.ref.force %clock, %en1, %w_ref, %val1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // Second force.
    firrtl.ref.force %clock, %en2, %w_ref, %val2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // Release (highest priority - last in chain).
    firrtl.ref.release %clock, %en2, %w_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // No `reset` port on this module, so the state-machine registers are plain
    // `firrtl.reg` (no reset value).
    // CHECK: %forced = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %forcedValue = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<8>
    // forceActive ORs the two force predicates before gating with forceWins.
    // CHECK-DAG: firrtl.or %en1, %en2
    // Override mux drives the target wire from the original driver (%c0) when unforced.
    // CHECK: firrtl.mux(%forced, %forcedValue, %{{.+}})
    // CHECK: firrtl.matchingconnect %w, %{{.+}}
  }
}

// -----
// TEST 3: Mix of clocked force and force_initial.  The initial access has no
// clock and rides along on the clocked access's clock (it is not a gated-clock
// root and does not error).

// CHECK-LABEL: firrtl.module @MixedClockedAndInitial
firrtl.circuit "MixedClockedAndInitial" {
  firrtl.module @MixedClockedAndInitial(in %clock: !firrtl.clock, in %en_clocked: !firrtl.uint<1>, in %en_initial: !firrtl.uint<1>, in %val1: !firrtl.uint<8>, in %val2: !firrtl.uint<8>) {
    // CHECK: %w = firrtl.wire : !firrtl.uint<8>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>

    // Clocked force.
    firrtl.ref.force %clock, %en_clocked, %w_ref, %val1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // Initial force (no clock).
    firrtl.ref.force_initial %en_initial, %w_ref, %val2 : !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // CHECK: %forced = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %forcedValue = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    // Both forces (clocked + initial) OR their predicates into forceActive.
    // CHECK-DAG: firrtl.or %en_clocked, %en_initial
    // The initial access carries no clock, so the state-machine registers run
    // on the clocked access's clock (%clock), used directly (no bundle wire).
  }
}

// -----
// TEST 4: Force and release target the same wire via DIFFERENT rwprobe SSA
// values (post-ExpandWhens shape).  The pass must produce one state machine
// with both forceActive and releaseActive wired correctly.
//
// %w_ref comes from firrtl.wire forceable (visited and mapped in probeToHWMap).
// %w_ref2 comes from an explicit firrtl.ref.rwprobe on the same inner sym
// (also visited and mapped to the same hw value in probeToHWMap).
//
// Expected output after fix:
//   - Exactly ONE forced register, ONE forcedValue register.
//   - forceActive driven by %force_en (not constant 0).
//   - releaseActive driven by a non-constant expression involving %release_en.

// CHECK-LABEL: firrtl.module @ForceReleaseSplitRWProbes
firrtl.circuit "ForceReleaseSplitRWProbes" {
  firrtl.module @ForceReleaseSplitRWProbes(
      in %clock:      !firrtl.clock,
      in %reset:      !firrtl.uint<1>,
      in %force_en:   !firrtl.uint<1>,
      in %release_en: !firrtl.uint<1>,
      in %val:        !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire sym @w_sym forceable :
        !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>

    // A second rwprobe value for the SAME inner symbol — this is exactly the
    // shape firrtl-expand-whens produces when force/release are in separate
    // when branches.
    %w_ref2 = firrtl.ref.rwprobe <@ForceReleaseSplitRWProbes::@w_sym> :
        !firrtl.rwprobe<uint<8>>

    // Force uses %w_ref; release uses %w_ref2.  Both target @w_sym.
    firrtl.ref.force %clock, %force_en, %w_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %release_en, %w_ref2 :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // Correct output assertions:

    // Exactly ONE forced register and ONE forcedValue register.
    // The bug produces two (forced + forced_0); a duplicate would be renamed
    // with a numeric suffix, so assert no such second register appears.
    // CHECK:     %forced = firrtl.regreset {{.+}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: %forced_{{[0-9]+}} = firrtl.regreset
    // CHECK:     %forcedValue = firrtl.regreset {{.+}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK-NOT: %forcedValue_{{[0-9]+}} = firrtl.regreset

    // forceActive must use %force_en in an AND gate (priority scheme, not constant 0).
    // CHECK-DAG: firrtl.and %force_en, %{{.+}}

    // releaseActive must be gated by NOT(forceWins) — both ops must appear.
    // CHECK-DAG: firrtl.not %{{.+}}
    // CHECK-DAG: firrtl.and %release_en, %{{.+}}

    // Exactly ONE override mux and ONE connect to %w, emitted after the
    // control-logic reduction (at the end of the block).
    // CHECK:     firrtl.mux(%{{.+}}, %{{.+}}, %{{.+}})
    // CHECK:     firrtl.matchingconnect %w, %{{.+}}
  }
}

// -----
// TEST 5: ForceRelease via split rwprobes on a REGISTER (regreset).
// Same bug scenario as Test 4 but targeting a regreset instead of a wire.
//
// Expected: one forced/forcedValue pair, forceActive driven by %force_en,
// releaseActive driven by expression containing %release_en.

// CHECK-LABEL: firrtl.module @ForceReleaseSplitRWProbesReg
firrtl.circuit "ForceReleaseSplitRWProbesReg" {
  firrtl.module @ForceReleaseSplitRWProbesReg(
      in %clock:      !firrtl.clock,
      in %reset:      !firrtl.uint<1>,
      in %next:       !firrtl.uint<8>,
      in %force_en:   !firrtl.uint<1>,
      in %release_en: !firrtl.uint<1>,
      in %val:        !firrtl.uint<8>) {
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    %r, %r_ref = firrtl.regreset sym @r_sym %clock, %reset, %c0 forceable :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>,
        !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %next : !firrtl.uint<8>

    // Second rwprobe for the same inner sym — different SSA value.
    %r_ref2 = firrtl.ref.rwprobe <@ForceReleaseSplitRWProbesReg::@r_sym> :
        !firrtl.rwprobe<uint<8>>

    firrtl.ref.force %clock, %force_en, %r_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %release_en, %r_ref2 :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // One state machine.
    // CHECK:     %forced = firrtl.regreset {{.+}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: %forced_{{[0-9]+}} = firrtl.regreset
    // forceActive uses %force_en in AND (priority scheme).
    // CHECK-DAG: firrtl.and %force_en, %{{.+}}
    // releaseActive has %release_en gated by NOT(forceWins).
    // CHECK-DAG: firrtl.and %release_en, %{{.+}}
    // Override mux replaces the existing next-value connect (appears after the
    // control-logic reduction, at the end of the block).
    // CHECK:     firrtl.mux(%{{.+}}, %{{.+}}, %next)
    // CHECK:     firrtl.matchingconnect %r, %{{.+}}
  }
}

// -----
// TEST 6: Sanity check — force AND release sharing the SAME %w_ref (the
// non-bug case) must continue to work and produce exactly one state machine.
// This tests that the fix does not break the common path.

// CHECK-LABEL: firrtl.module @ForceReleaseSameRWProbe
firrtl.circuit "ForceReleaseSameRWProbe" {
  firrtl.module @ForceReleaseSameRWProbe(
      in %clock:      !firrtl.clock,
      in %force_en:   !firrtl.uint<1>,
      in %release_en: !firrtl.uint<1>,
      in %val:        !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable :
        !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>

    // Both ops use the same %w_ref — the normal (non-bug) path.
    firrtl.ref.force %clock, %force_en, %w_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %release_en, %w_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // CHECK:     %forced = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %forced_{{[0-9]+}} = firrtl.reg
    // CHECK:     %forcedValue = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<8>
    // CHECK-DAG: firrtl.and %force_en, %{{.+}}
    // CHECK-DAG: firrtl.not %{{.+}}
    // CHECK-DAG: firrtl.and %release_en, %{{.+}}
    // CHECK-DAG: firrtl.matchingconnect %w, %{{.+}}
  }
}

// -----

// TEST 7: A forceable RWProbe exported by a module that is instantiated more
// than once.
// The `_force_ctrl` port is added to the module and to *every* instance; the
// module's port count and its instances' port counts must stay in lockstep.
// This is done in a sequential post-pass (the per-module conversion runs in
// parallel), so a module's signature and all of its instances are mutated
// together.

// CHECK-LABEL: firrtl.module @Child
// The child gets exactly one inbound control port.
// CHECK-SAME: in %probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>

firrtl.circuit "MultiInst" {
  firrtl.module @Child(out %probe_out: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %probe_out, %w_ref : !firrtl.rwprobe<uint<8>>
  }

  // CHECK-LABEL: firrtl.module @MultiInst
  firrtl.module @MultiInst(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // Both instances must carry the matching `_force_ctrl` input port and each
    // is driven by its own forwarding control wire.
    // CHECK: firrtl.instance a @Child(out probe_out: !firrtl.uint<8>, in probe_out_force_ctrl: !firrtl.bundle<{{.*}}>)
    // CHECK: firrtl.matchingconnect %a_probe_out_force_ctrl, %{{.+}}
    // CHECK: firrtl.instance b @Child(out probe_out: !firrtl.uint<8>, in probe_out_force_ctrl: !firrtl.bundle<{{.*}}>)
    // CHECK: firrtl.matchingconnect %b_probe_out_force_ctrl, %{{.+}}
    %a_probe = firrtl.instance a @Child(out probe_out: !firrtl.rwprobe<uint<8>>)
    %b_probe = firrtl.instance b @Child(out probe_out: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.force %clock, %en, %a_probe, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.force %clock, %en, %b_probe, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}


// -----
// TEST 8: Verify reset VALUE is 0 for `forced` (uint<1>) and a typed zero for
// `forcedValue` (uint<16>, non-trivial width to catch width bugs).
//
// CHECK-LABEL: firrtl.module @ForceResetValueIsZero
firrtl.circuit "ForceResetValueIsZero" {
  firrtl.module @ForceResetValueIsZero(
      in %clock:  !firrtl.clock,
      in %reset:  !firrtl.uint<1>,
      in %enable: !firrtl.uint<1>,
      in %val:    !firrtl.uint<16>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<16>, !firrtl.rwprobe<uint<16>>
    %c0 = firrtl.constant 0 : !firrtl.uint<16>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<16>

    firrtl.ref.force %clock, %enable, %w_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<16>>, !firrtl.uint<16>

    // The reset values must be zero FIRRTL constants.
    // CHECK-DAG: firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-DAG: firrtl.constant 0 : !firrtl.uint<16>
    // `forced` resets to uint<1> value 0.
    // CHECK: %forced = firrtl.regreset {{.+}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    // `forcedValue` resets to uint<16> value 0.
    // CHECK: %forcedValue = firrtl.regreset {{.+}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<16>, !firrtl.uint<16>
  }
}


// -----
// TEST 9: Three-level hierarchy with a register at the leaf and forces from
// Leaf owns the forceable register and forces it locally; Middle and Top each
// force the same probe through the instance chain.  Each module in the chain
// gets a `_force_ctrl` port and merges its local force with the inbound one.

// CHECK-LABEL: firrtl.circuit "Middle"
firrtl.circuit "Middle" {
  // Leaf module with forceable register
  // CHECK: firrtl.module @Leaf(out %reg_probe: !firrtl.uint<8>
  // The leaf gains an inbound control port for the exported probe.
  // CHECK-SAME: in %reg_probe_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>
  firrtl.module @Leaf(out %reg_probe: !firrtl.rwprobe<uint<8>>, in %clock: !firrtl.clock, in %data_in: !firrtl.uint<8>, in %enable: !firrtl.uint<1>) {
    %reg, %reg_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %reg, %data_in : !firrtl.uint<8>
    firrtl.ref.define %reg_probe, %reg_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %enable, %reg_ref, %data_in : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // Leaf builds a state machine for its local register and overrides its
    // next-value connect with the override mux.
    // CHECK: %forced = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %forcedValue = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<8>
    // CHECK: firrtl.mux(%forced, %forcedValue, %{{.+}})
    // CHECK: firrtl.matchingconnect %reg, %{{.+}}
    // The local force (%enable) is merged with the inbound control port
    // (local force wins), so forceActive ORs the local predicate with inbound.
    // CHECK: firrtl.or %enable, %{{.+}}
    // CHECK: firrtl.mux(%enable, %{{.+}}, %{{.+}})
  }

  // Middle level module that instantiates Leaf and can force the register
  // CHECK-LABEL: firrtl.module @Middle
  // Middle gains its own inbound control port and drives the Leaf instance's
  // control port from its forwarding wire.
  // CHECK-SAME: in %reg_probe_out_force_ctrl: !firrtl.bundle
  // CHECK: firrtl.instance leaf @Leaf(
  // CHECK-SAME: in reg_probe_force_ctrl: !firrtl.bundle
  // CHECK: firrtl.matchingconnect %leaf_reg_probe_force_ctrl, %{{.+}}
  // Middle's local force (%enable_middle) is merged with its inbound port.
  // CHECK: firrtl.or %enable_middle, %{{.+}}
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

  // Top level module that instantiates Middle and can also force the register
  // CHECK-LABEL: firrtl.module @ThreeLevelHierarchy
  firrtl.module @ThreeLevelHierarchy(in %clock: !firrtl.clock, in %data_in: !firrtl.uint<8>, in %enable_middle: !firrtl.uint<1>, in %value_middle: !firrtl.uint<8>, in %enable_top: !firrtl.uint<1>, in %value_top: !firrtl.uint<8>) {
    %middle_probe, %middle_clock, %middle_data, %middle_enable, %middle_value = firrtl.instance middle @Middle(out reg_probe_out: !firrtl.rwprobe<uint<8>>, in clock: !firrtl.clock, in data_in: !firrtl.uint<8>, in enable_middle: !firrtl.uint<1>, in value_middle: !firrtl.uint<8>)
    firrtl.matchingconnect %middle_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %middle_data, %data_in : !firrtl.uint<8>
    firrtl.matchingconnect %middle_enable, %enable_middle : !firrtl.uint<1>
    firrtl.matchingconnect %middle_value, %value_middle : !firrtl.uint<8>

    // Force from top level - same register probe that middle level forces
    firrtl.ref.force %clock, %enable_top, %middle_probe, %value_top : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // The top-level force drives the Middle instance's inbound control port via
    // a forwarding wire; Top itself is not exported so it needs no control port.
    // CHECK: firrtl.instance middle @Middle(
    // CHECK-SAME: in reg_probe_out_force_ctrl: !firrtl.bundle
    // CHECK: firrtl.matchingconnect %middle_reg_probe_out_force_ctrl, %{{.+}}
    // forceActive on the forwarding wire is driven from %enable_top.
    // CHECK: firrtl.matchingconnect %{{.+}}, %enable_top : !firrtl.uint<1>
  }
}


// -----
// TEST 10: Force on a RegOp (plain register, no reset) with a release.
// The existing connect on the register (next-value connect) must be found
// and replaced by the override mux; no second driver.

// CHECK-LABEL: firrtl.circuit "PlainRegForceRelease"
firrtl.circuit "PlainRegForceRelease" {
  firrtl.module @PlainRegForceRelease(
      in %clock: !firrtl.clock,
      in %next: !firrtl.uint<8>,
      in %en_force: !firrtl.uint<1>,
      in %en_release: !firrtl.uint<1>,
      in %val: !firrtl.uint<8>) {
    %r, %r_ref = firrtl.reg %clock forceable :
        !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %next : !firrtl.uint<8>

    firrtl.ref.force %clock, %en_force, %r_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %en_release, %r_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // Existing next-value connect is replaced by the override mux.
    // CHECK: %r = firrtl.reg %clock
    // Release gated by !forceWins.
    // CHECK-DAG: %[[NFW:.+]] = firrtl.not %{{.+}}
    // CHECK-DAG: firrtl.and %en_release, %[[NFW]]
    // Override mux: mux(forced, forcedValue, next), emitted at end of block.
    // CHECK: firrtl.mux(%{{.+}}, %{{.+}}, %next)
    // Exactly ONE connect to %r (the override mux output).
    // CHECK: firrtl.matchingconnect %r, %{{.+}}
  }
}

// -----
// TEST 11: Multiple releases with a single force.
// releaseActive = OR(all release preds) AND NOT(forceWins).
// forceActive = OR(force preds) AND forceWins.

// CHECK-LABEL: firrtl.circuit "MultipleReleasesSingleForce"
firrtl.circuit "MultipleReleasesSingleForce" {
  firrtl.module @MultipleReleasesSingleForce(
      in %clock: !firrtl.clock,
      in %en_f: !firrtl.uint<1>,
      in %en_r1: !firrtl.uint<1>,
      in %en_r2: !firrtl.uint<1>,
      in %en_r3: !firrtl.uint<1>,
      in %val: !firrtl.uint<8>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>

    firrtl.ref.force %clock, %en_f, %w_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %en_r1, %w_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
    firrtl.ref.release %clock, %en_r2, %w_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
    firrtl.ref.release %clock, %en_r3, %w_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // releaseActive = (en_r1 OR en_r2 OR en_r3) AND NOT(forceWins).
    // CHECK-DAG: firrtl.or %en_r1, %en_r2
    // CHECK-DAG: firrtl.or %{{.+}}, %en_r3
    // forceActive = en_f AND forceWins.
    // CHECK-DAG: firrtl.and %en_f, %{{.+}}
  }
}

// -----

// TEST 12: This test verifies selective force behavior: a parent module
// instantiates a child module twice, forcing the RWProbe from only one instance
// while merely reading from the other instance's probe (no force/release).

// CHECK-LABEL: firrtl.circuit "SelectiveForce"
firrtl.circuit "SelectiveForce" {
  // Child module exports the RWProbe and gains an inbound control port
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

    // Instance 'b' is NOT forced (only read), so its control wire is initialized with defaults
    // CHECK-NEXT: %[[FALSE:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-NEXT: %[[INVALID:.+]] = firrtl.invalidvalue : !firrtl.uint<8>
    // CHECK-NEXT: %[[B_CTRL:.+]] = firrtl.wire : !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>
    // CHECK-NEXT: %[[B_FA:.+]] = firrtl.subfield %[[B_CTRL]][forceActive]
    // CHECK-NEXT: %[[B_RA:.+]] = firrtl.subfield %[[B_CTRL]][releaseActive]
    // CHECK-NEXT: %[[B_FV:.+]] = firrtl.subfield %[[B_CTRL]][forcedValue]
    // CHECK-NEXT: %[[INVALID2:.+]] = firrtl.invalidvalue : !firrtl.uint<8>
    // CHECK-NEXT: firrtl.matchingconnect %[[B_FA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[B_RA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[B_FV]], %[[INVALID2]]
    // Instance 'a' is forced, so its control wire is driven by the force operation
    // CHECK-NEXT: %[[A_CTRL:.+]] = firrtl.wire : !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>

    // CHECK-NEXT: %a_probe_out, %a_probe_out_force_ctrl = firrtl.instance a @Child(out probe_out: !firrtl.uint<8>, in probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>)
    // CHECK-NEXT: firrtl.matchingconnect %a_probe_out_force_ctrl, %[[A_CTRL]]
    %a_probe = firrtl.instance a @Child(out probe_out: !firrtl.rwprobe<uint<8>>)

    // CHECK-NEXT: %b_probe_out, %b_probe_out_force_ctrl = firrtl.instance b @Child(out probe_out: !firrtl.uint<8>, in probe_out_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>)
    // CHECK-NEXT: firrtl.matchingconnect %b_probe_out_force_ctrl, %[[B_CTRL]]
    %b_probe = firrtl.instance b @Child(out probe_out: !firrtl.rwprobe<uint<8>>)

    firrtl.ref.force %clock, %enable, %a_probe, %force_value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // Instance 'b' is only read, not forced
    // CHECK: firrtl.matchingconnect %read_value, %b_probe_out
    %b_read = firrtl.ref.resolve %b_probe : !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %read_value, %b_read : !firrtl.uint<8>

    // The force operation drives instance 'a's control wire
    // CHECK: %[[A_FA:.+]] = firrtl.subfield %[[A_CTRL]][forceActive]
    // CHECK-NEXT: %[[A_RA:.+]] = firrtl.subfield %[[A_CTRL]][releaseActive]
    // CHECK-NEXT: %[[A_FV:.+]] = firrtl.subfield %[[A_CTRL]][forcedValue]
    // CHECK-NEXT: %[[A_CLK:.+]] = firrtl.subfield %[[A_CTRL]][clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[A_FA]], %enable
    // CHECK-NEXT: firrtl.matchingconnect %[[A_RA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[A_FV]], {{%.+}}
    // CHECK-NEXT: firrtl.matchingconnect %[[A_CLK]], %clock
  }
}