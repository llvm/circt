// RUN: circt-opt --firrtl-probes-to-signals --cse --split-input-file %s | FileCheck %s

// This test file covers force/release synthesis for the ProbesToSignals pass.
// The pass synthesizes force/release operations into a per-probe state machine:
// - a `forced` register (UInt<1>) tracking whether the target is forced,
// - for several forces, a one-hot `forceWinner` latch recording *which* force is
//   in effect; the forced value itself is never registered, so a forced target
//   keeps tracking the winning force's RHS the way Verilog `force a = v` does,
// - an override mux injected on every *read* of the target, published through a
//   `<target>_forced` wire; the target's own driver (a wire's connect, a
//   register's next state and reset) is left completely untouched, and
// - a control bundle wire {forceActive, releaseActive, forcedValue, clk} whose
//   fields are driven exactly once by the priority reduction.
//
// A target that nothing reads gets no override at all: there is nothing to
// observe the force through, so the state machine is left dead.
//
// Test scenarios covered:
//  1. Force/release of a register in the same module.
//  2. Multiple force/release operations to the same target (priority ordering).
//  3. Force/release targeting the same wire via DIFFERENT rwprobe SSA values
//     (post-ExpandWhens shape) collapse into one state machine.
//  4. Same split-rwprobe scenario on a register.
//  5. Force + release sharing the SAME rwprobe SSA value (non-bug path).
//  6. A probe exported by a module instantiated more than once (lockstep ports).
//  7. Preset (power-on) values of the state-machine registers are typed zeros.
//  8. Three-level hierarchy: forces from leaf, middle, and top compose.
//  9. Force + release on a plain (no-reset) register.
// 10. Multiple releases reduced against a single force.
// 11. Un-forced RWProbe.
// 12. Force of an instance probe through a same-type `ref.cast`.
// 13. Force/release of a local target through a same-type `ref.cast`.
// 14. Export (`ref.define`) of a local target through a same-type `ref.cast`.
// 15. Pure re-export (no local force) must not tie off the forwarding wire.
// 16. Two probes on one instance, only one forced (per-result tie-off).
// 17. Release-only (no force at all) local target.
// 18. Release-only of an instance probe (forwarded to the child's control).
// 19. Release-only of an exported target (local release merged with inbound).
// 20. Release-only of an instance probe through a same-type `ref.cast`.
// 21. Self-referential register next state (`r <= r + 1`) plus a force.
// 24. A forced target nobody reads gets no override.
// 25. Three clocked forces: the sticky one-hot winner keeps the *live* RHS of
//     whichever force is in effect (a plain last-wins mux would collapse back
//     to the first force once every predicate dropped).
// 26. Force of a register clocked by a gated clock.
// 27. Force of a child's register clocked by a gated clock local to the child.


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

    // The register keeps its own next-state driver: the force must not become
    // part of the flop equation (it would land a cycle late and lose to reset).
    // CHECK: firrtl.matchingconnect %r, %c1_ui8
    // Readers observe the override instead.
    // CHECK: firrtl.matchingconnect %o, %r_forced
    // forceActive is forceWins itself (the winning predicate already implies
    // a force is active).
    // releaseActive is gated by !forceWins so a concurrent force suppresses it.
    // CHECK-DAG: %[[NFW:.+]] = firrtl.not %{{.+}}
    // CHECK-DAG: firrtl.and %release, %[[NFW]]
    // The only state is the `forced` flag: a plain `firrtl.reg` with an
    // `initial` attribute of 0 (a power-on value that prevents X-initialization
    // without depending on the module's `reset` port firing).  The forced
    // *value* is not registered -- a single force needs no state at all, its
    // RHS is read live.  It is emitted after the existing target logic.
    // CHECK: %{{.+}} = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<8>
    // The override reads the raw register and publishes the observed value,
    // muxing in the force's live RHS (`%value`, not a snapshot of it).
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

    // First force (lowest priority - earliest in PriorityMux chain).
    firrtl.ref.force %clock, %en1, %w_ref, %val1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // Second force.
    firrtl.ref.force %clock, %en2, %w_ref, %val2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // Release (highest priority - last in chain).
    firrtl.ref.release %clock, %en2, %w_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // The wire keeps its own single driver.
    // CHECK: firrtl.matchingconnect %w, %c0_ui8
    // CHECK: firrtl.matchingconnect %o, %w_forced
    // forceActive is forceWins itself, the priority mux chain over the two
    // force predicates and the release predicate.
    // CHECK: %[[FW0:.+]] = firrtl.mux(%en2, %c1_ui1, %en1)
    // CHECK: %[[FA:.+]] = firrtl.mux(%en2, %c0_ui1, %[[FW0]])
    // Two forces, so which one is in effect is latched (one bit per force after
    // the first) while both RHS values stay live.  `en2` is the last force, so
    // no later predicate masks it.
    // CHECK: %forceWinner = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[FV:.+]] = firrtl.mux(%forceWinner, %val2, %val1)
    // CHECK: %[[WIN:.+]] = firrtl.mux(%[[FA]], %en2, %forceWinner)
    // CHECK: firrtl.matchingconnect %forceWinner, %[[WIN]]
    // The state-machine registers are plain `firrtl.reg` with an `initial`
    // power-on value of 0.  The `forced` register is emitted after the winner
    // state because both are materialized at module end.
    // CHECK: %{{.+}} = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %{{.+}}_{{[0-9]+}} = firrtl.reg
    // The override falls back to the wire (and hence its own driver) when
    // unforced.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[FV]], %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 3: Force and release target the same wire via DIFFERENT rwprobe SSA
// values (post-ExpandWhens shape).  The pass must produce one state machine
// with both forceActive and releaseActive wired correctly.
//
// %w_ref comes from firrtl.wire forceable (visited and mapped in probeToHWMap).
// %w_ref2 comes from an explicit firrtl.ref.rwprobe on the same inner sym
// (also visited and mapped to the same hw value in probeToHWMap).
//
// Expected output after fix:
//   - Exactly ONE forced register.
//   - forceActive driven by %force_en (not constant 0).
//   - releaseActive driven by a non-constant expression involving %release_en.

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

    // Exactly ONE observed wire and ONE forced register.  The bug produces two
    // (forced + forced_0); a duplicate would be renamed with a numeric suffix,
    // so assert no such second one appears.
    // CHECK:     %w_forced = firrtl.wire : !firrtl.uint<8>
    // CHECK-NOT: %w_forced_{{[0-9]+}} = firrtl.wire
    // %w keeps its single original driver; the reader sees the observed value.
    // CHECK:     firrtl.matchingconnect %w, %c0_ui8
    // CHECK:     firrtl.matchingconnect %o, %w_forced

    // forceActive is forceWins itself, driven by %force_en through the
    // priority mux (not constant 0).
    // CHECK-DAG: firrtl.mux(%release_en, %{{.+}}, %force_en)

    // releaseActive must be gated by NOT(forceWins) — both ops must appear.
    // CHECK-DAG: firrtl.not %{{.+}}
    // CHECK-DAG: firrtl.and %release_en, %{{.+}}

    // The generated state is emitted after the original driver and reduction.
    // CHECK:     %{{.+}} = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %{{.+}}_{{[0-9]+}} = firrtl.reg

    // Exactly ONE override, emitted after the control-logic reduction (at the
    // end of the block).
    // CHECK:     %[[OVR:.+]] = firrtl.mux(%forced, %val, %w)
    // CHECK:     firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 4: ForceRelease via split rwprobes on a REGISTER (regreset).
// Same bug scenario as Test 3 but targeting a regreset instead of a wire.
//
// Expected: one forced register, forceActive driven by %force_en,
// releaseActive driven by expression containing %release_en.

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

    // Second rwprobe for the same inner sym — different SSA value.
    %r_ref2 = firrtl.ref.rwprobe <@ForceReleaseSplitRWProbesReg::@r_sym> :
        !firrtl.rwprobe<uint<8>>

    firrtl.ref.force %clock, %force_en, %r_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %release_en, %r_ref2 :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // One state machine.
    // CHECK:     %r_forced = firrtl.wire : !firrtl.uint<8>
    // CHECK-NOT: %r_forced_{{[0-9]+}} = firrtl.wire
    // The register keeps its own next-value connect and its reset.
    // CHECK:     firrtl.matchingconnect %r, %next
    // CHECK:     firrtl.matchingconnect %o, %r_forced
    // forceActive is forceWins, driven by %force_en through the priority mux.
    // CHECK-DAG: firrtl.mux(%release_en, %{{.+}}, %force_en)
    // releaseActive has %release_en gated by NOT(forceWins).
    // CHECK-DAG: firrtl.and %release_en, %{{.+}}
    // The generated state is emitted after the original register logic.
    // CHECK:     %{{.+}} = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %{{.+}}_{{[0-9]+}} = firrtl.reg
    // The override reads the raw register (appears after the control-logic
    // reduction, at the end of the block).
    // CHECK:     %[[OVR:.+]] = firrtl.mux(%forced, %val, %r)
    // CHECK:     firrtl.matchingconnect %r_forced, %[[OVR]]
  }
}

// -----
// TEST 5: Sanity check — force AND release sharing the SAME %w_ref (the
// non-bug case) must continue to work and produce exactly one state machine.
// This tests that the fix does not break the common path.

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

    // Both ops use the same %w_ref — the normal (non-bug) path.
    firrtl.ref.force %clock, %force_en, %w_ref, %val :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %clock, %release_en, %w_ref :
        !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // CHECK-DAG: firrtl.mux(%release_en, %{{.+}}, %force_en)
    // CHECK-DAG: firrtl.not %{{.+}}
    // CHECK-DAG: firrtl.and %release_en, %{{.+}}
    // CHECK-DAG: firrtl.matchingconnect %w, %c0_ui8
    // The state register is emitted after the target and control reduction.
    // CHECK:     %{{.+}} = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %{{.+}}_{{[0-9]+}} = firrtl.reg
    // CHECK:     firrtl.matchingconnect %w_forced, %{{.+}}
  }
}

// -----

// TEST 6: A forceable RWProbe exported by a module that is instantiated more
// than once.
// The probe port becomes a `{data, ctrl}` bundle on the module and on *every*
// instance; the module's port count and its instances' port counts must stay
// in lockstep.

// CHECK-LABEL: firrtl.module @Child
// The child's probe port becomes a {data, ctrl} bundle.
// CHECK-SAME: out %probe_out: !firrtl.bundle<data: uint<8>, ctrl flip: bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>>

firrtl.circuit "MultiInst" {
  firrtl.module @Child(out %probe_out: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %probe_out, %w_ref : !firrtl.rwprobe<uint<8>>
  }

  // CHECK-LABEL: firrtl.module @MultiInst
  firrtl.module @MultiInst(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // Both instances must carry the matching bundled probe port and each
    // control subfield is driven from the instance.
    // CHECK: firrtl.instance a @Child(out probe_out: !firrtl.bundle<{{.*}}>)
    // CHECK: firrtl.instance b @Child(out probe_out: !firrtl.bundle<{{.*}}>)
    // CHECK: firrtl.matchingconnect %{{.+}}, %en : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %{{.+}}, %en : !firrtl.uint<1>
    %a_probe = firrtl.instance a @Child(out probe_out: !firrtl.rwprobe<uint<8>>)
    %b_probe = firrtl.instance b @Child(out probe_out: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.force %clock, %en, %a_probe, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.force %clock, %en, %b_probe, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}


// -----
// TEST 7: Verify the `initial` (power-on) VALUE is 0 for `forced`, and that the
// forced value itself is *not* registered: a wide probed type (uint<16>) must
// not produce a register at all, the force's RHS being read live.
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

    // The register carries a typed-zero `initial` power-on value: `forced`
    // initializes to a uint<1> value 0.
    // CHECK: %forced = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // Nothing of the probed type is registered.
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<16>
  }
}


// -----
// TEST 8: Three-level hierarchy with a register at the leaf.  Leaf owns the
// forceable register and forces it locally; Middle and Top each force the same
// probe through the instance chain.  Each module in the chain merges its local
// force with the one arriving on its probe port's ctrl field.

// CHECK-LABEL: firrtl.circuit "Middle"
firrtl.circuit "Middle" {
  // Leaf module with forceable register
  // The leaf's exported probe becomes a {data, ctrl} bundle port.
  // CHECK: firrtl.module @Leaf(out %reg_probe: !firrtl.bundle<data: uint<8>, ctrl flip: bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>>
  firrtl.module @Leaf(out %reg_probe: !firrtl.rwprobe<uint<8>>, in %clock: !firrtl.clock, in %data_in: !firrtl.uint<8>, in %enable: !firrtl.uint<1>) {
    %reg, %reg_ref = firrtl.reg %clock forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %reg, %data_in : !firrtl.uint<8>
    firrtl.ref.define %reg_probe, %reg_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %enable, %reg_ref, %data_in : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // Leaf builds a state machine for its local register and routes the reads of
    // it through the override.
    // The bundled probe port is split into its data and ctrl subfields.
    // CHECK: %[[DATA:.+]] = firrtl.subfield %reg_probe[data]
    // CHECK: %[[CTRL:.+]] = firrtl.subfield %reg_probe[ctrl]
    // CHECK: %reg_forced = firrtl.wire : !firrtl.uint<8>
    // The register keeps its own next-value connect; the exported probe carries
    // the observed value.
    // CHECK: firrtl.matchingconnect %reg, %data_in
    // CHECK: firrtl.matchingconnect %[[DATA]], %reg_forced
    // The local force (%enable) is merged with the ctrl subfield's forceActive.
    // CHECK: %[[CTRL_FORCE:.+]] = firrtl.subfield %[[CTRL]][forceActive]
    // CHECK: %[[CTRL_RELEASE:.+]] = firrtl.subfield %[[CTRL]][releaseActive]
    // CHECK: %[[CTRL_VALUE:.+]] = firrtl.subfield %[[CTRL]][forcedValue]
    // CHECK: %[[ANY_FORCE:.+]] = firrtl.or %enable, %[[CTRL_FORCE]]
    // ... and which side forced last is latched, so the merged value keeps
    // tracking that side's live RHS after both predicates drop.
    // CHECK: %forcedByLocal = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[MERGED:.+]] = firrtl.mux(%forcedByLocal, %data_in, %[[CTRL_VALUE]])
    // CHECK: %[[LOCAL_NEXT:.+]] = firrtl.mux(%[[ANY_FORCE]], %enable, %forcedByLocal)
    // CHECK: firrtl.matchingconnect %forcedByLocal, %[[LOCAL_NEXT]]
    // The outer state register is emitted after the local winner state.
    // CHECK: %{{.+}} = firrtl.reg {{.+}} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[MERGED]], %reg)
    // CHECK: firrtl.matchingconnect %reg_forced, %[[OVR]]
  }

  // Middle level module that instantiates Leaf and can force the register
  // CHECK-LABEL: firrtl.module @Middle
  // Middle's exported probe becomes a {data, ctrl} bundle port, and it
  // instantiates Leaf with the same bundled probe port shape.
  // CHECK-SAME: out %reg_probe_out: !firrtl.bundle<data: uint<8>, ctrl flip: bundle
  // CHECK: firrtl.instance leaf @Leaf(
  // CHECK-SAME: out reg_probe: !firrtl.bundle<data: uint<8>, ctrl flip: bundle
  // Middle's local force (%enable_middle) is merged with the leaf's inbound
  // control fields.
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
    // The top-level force drives the Middle instance's bundled probe port's
    // ctrl subfields directly; Top itself is not exported so it needs no
    // extra control port.
    // CHECK: firrtl.instance middle @Middle(
    // CHECK-SAME: out reg_probe_out: !firrtl.bundle<data: uint<8>, ctrl flip: bundle
    // forceActive on the ctrl subfield is driven from %enable_top.
    // CHECK: firrtl.matchingconnect %{{.+}}, %enable_top : !firrtl.uint<1>
  }
}


// -----
// TEST 9: Force on a RegOp (plain register, no reset) with a release.
// The existing next-value connect on the register stays put; the override lands
// on the reads.  No second driver of %r.

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
    // Exactly ONE connect to %r, the original next-value connect.
    // CHECK: firrtl.matchingconnect %r, %next
    // CHECK-NOT: firrtl.matchingconnect %r, %
    // CHECK: firrtl.matchingconnect %o, %r_forced
    // Release gated by !forceWins.
    // CHECK-DAG: %[[NFW:.+]] = firrtl.not %{{.+}}
    // CHECK-DAG: firrtl.and %en_release, %[[NFW]]
    // Override mux: mux(forced, val, r), emitted at end of block.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %val, %r)
    // CHECK: firrtl.matchingconnect %r_forced, %[[OVR]]
  }
}

// -----
// TEST 10: Multiple releases with a single force.
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
    // forceActive is forceWins itself.
  }
}

// -----

// TEST 11: This test verifies selective force behavior: a parent module
// instantiates a child module twice, forcing the RWProbe from only one instance
// while merely reading from the other instance's probe (no force/release).

// CHECK-LABEL: firrtl.circuit "SelectiveForce"
firrtl.circuit "SelectiveForce" {
  // Child module exports the RWProbe as a bundled {data, ctrl} port.
  // CHECK: firrtl.module @Child(out %probe_out: !firrtl.bundle<data: uint<8>, ctrl flip: bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>>)
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

    // Both instances carry the bundled probe port; the ctrl subfield is
    // extracted from each instance result directly (no forwarding wire).
    // CHECK-NEXT: %a_probe_out = firrtl.instance a @Child(out probe_out: !firrtl.bundle<data: uint<8>, ctrl flip: bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>>)
    // CHECK-NEXT: %[[A_CTRL:.+]] = firrtl.subfield %a_probe_out[ctrl]
    %a_probe = firrtl.instance a @Child(out probe_out: !firrtl.rwprobe<uint<8>>)

    // CHECK-NEXT: %b_probe_out = firrtl.instance b @Child(out probe_out: !firrtl.bundle<data: uint<8>, ctrl flip: bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>>)
    // CHECK-NEXT: %[[B_DATA:.+]] = firrtl.subfield %b_probe_out[data]
    // CHECK-NEXT: %[[B_CTRL:.+]] = firrtl.subfield %b_probe_out[ctrl]
    %b_probe = firrtl.instance b @Child(out probe_out: !firrtl.rwprobe<uint<8>>)

    firrtl.ref.force %clock, %enable, %a_probe, %force_value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // Instance 'b' is only read, not forced
    // CHECK: firrtl.matchingconnect %read_value, %[[B_DATA]]
    %b_read = firrtl.ref.resolve %b_probe : !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %read_value, %b_read : !firrtl.uint<8>

    // Instance 'a' is forced, so its ctrl subfields are driven from the
    // reduced local control: the force predicate on forceActive, its RHS on
    // forcedValue.
    // CHECK: %[[FALSE:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[A_FA:.+]] = firrtl.subfield %[[A_CTRL]][forceActive]
    // CHECK-NEXT: %[[A_RA:.+]] = firrtl.subfield %[[A_CTRL]][releaseActive]
    // CHECK-NEXT: %[[A_FV:.+]] = firrtl.subfield %[[A_CTRL]][forcedValue]
    // CHECK-NEXT: %[[A_CLK:.+]] = firrtl.subfield %[[A_CTRL]][clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[A_FA]], %enable
    // CHECK-NEXT: firrtl.matchingconnect %[[A_RA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[A_FV]], %force_value
    // CHECK-NEXT: firrtl.matchingconnect %[[A_CLK]], %clock

    // Instance 'b' is NOT forced (only read), so its ctrl subfields get
    // inactive defaults, including a constant-0 clock so the child's SM is
    // not left with an undriven `clk` (which would X the unforced target).
    // CHECK: %[[ZEROCLK:.+]] = firrtl.specialconstant 0 : !firrtl.clock
    // CHECK: %[[B_FA:.+]] = firrtl.subfield %[[B_CTRL]][forceActive]
    // CHECK-NEXT: %[[B_RA:.+]] = firrtl.subfield %[[B_CTRL]][releaseActive]
    // CHECK-NEXT: %[[B_FV:.+]] = firrtl.subfield %[[B_CTRL]][forcedValue]
    // CHECK-NEXT: %[[B_CLK:.+]] = firrtl.subfield %[[B_CTRL]][clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[B_FA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[B_RA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[B_FV]], %{{.+}}
    // CHECK-NEXT: firrtl.matchingconnect %[[B_CLK]], %[[ZEROCLK]]
  }
}

// -----
// TEST 12: Force an instance probe through a same-type `ref.cast`.
// The cast must not invent a dummy wire / local SM: the force is forwarded
// to the child's inbound ctrl field, same as a direct force of the
// instance result.

// CHECK-LABEL: firrtl.module @CastChild
// CHECK-SAME: out %probe_out: !firrtl.bundle<data: uint<8>, ctrl flip: bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>>
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
    // CHECK: %c_probe_out = firrtl.instance c @CastChild(out probe_out: !firrtl.bundle<data: uint<8>, ctrl flip: bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>>)
    // CHECK-NEXT: %[[C_CTRL:.+]] = firrtl.subfield %c_probe_out[ctrl]
    %c_probe = firrtl.instance c @CastChild(out probe_out: !firrtl.rwprobe<uint<8>>)
    %cast = firrtl.ref.cast %c_probe : (!firrtl.rwprobe<uint<8>>) -> !firrtl.rwprobe<uint<8>>
    firrtl.ref.force %clock, %enable, %cast, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>

    // The same-type cast must not lower to a copy wire (which would be the
    // target of a parent-local state machine, leaving the child unforced), and
    // the control fields must not be tied off inactive (which would be a
    // second driver of these fields as well as dropping the force).
    // CHECK-NOT: firrtl.wire : !firrtl.uint<8>
    // CHECK-NOT: firrtl.specialconstant
    // CHECK-NOT: %forced = firrtl.reg
    // The force drives the child's ctrl subfields instead.
    // CHECK: %[[C_FA:.+]] = firrtl.subfield %[[C_CTRL]][forceActive]
    // CHECK-NEXT: %[[C_RA:.+]] = firrtl.subfield %[[C_CTRL]][releaseActive]
    // CHECK-NEXT: %[[C_FV:.+]] = firrtl.subfield %[[C_CTRL]][forcedValue]
    // CHECK-NEXT: %[[C_CLK:.+]] = firrtl.subfield %[[C_CTRL]][clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[C_FA]], %enable
    // CHECK-NEXT: firrtl.matchingconnect %[[C_RA]], %[[FALSE:.+]]
    // CHECK-NEXT: firrtl.matchingconnect %[[C_FV]], {{%.+}}
    // CHECK-NEXT: firrtl.matchingconnect %[[C_CLK]], %clock
  }
}
// -----
// TEST 13: Force + release of a LOCAL forceable target through a same-type
// `ref.cast`.  The cast must not lower to a copy wire: the state machine and
// the override mux must land on the real wire, exactly as for a direct force.

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

    // No copy wire for the cast, and no control bundle at all (a purely-local
    // target drives its registers from SSA).
    // CHECK-NOT: firrtl.wire
    // forceActive is forceWins itself; releaseActive is gated by !forceWins.
    // CHECK-DAG: firrtl.and %rel, %{{.+}}
    // The state register is emitted after the control reduction.
    // CHECK: %{{.+}} = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<8>
    // The override reads the real target, not a copy, and drives the force's
    // live RHS.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %value, %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 14: A child exports its forceable probe through a same-type `ref.cast`
// (`ref.define %port, %cast`).  The exported target must be resolved through
// the cast, so the child's state machine overrides the real wire and a force
// arriving on the probe port's ctrl field takes effect.

// CHECK-LABEL: firrtl.module @ExportCastChild
// CHECK-SAME: out %probe_out: !firrtl.bundle<data: uint<8>, ctrl flip: bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>>
firrtl.circuit "ForceExportedThroughCast" {
  firrtl.module @ExportCastChild(out %probe_out: !firrtl.rwprobe<uint<8>>) {
    // The bundled probe port is split into its data and ctrl subfields.
    // CHECK: %[[DATA:.+]] = firrtl.subfield %probe_out[data]
    // CHECK: %[[CTRL:.+]] = firrtl.subfield %probe_out[ctrl]
    // CHECK: %w = firrtl.wire : !firrtl.uint<8>
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    %cast = firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<8>>) -> !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %probe_out, %cast : !firrtl.rwprobe<uint<8>>

    // The state machine (and the override) is on %w, and %w -- not a copy of it
    // -- is what feeds the port.
    // CHECK: %w_forced = firrtl.wire : !firrtl.uint<8>
    // CHECK: firrtl.matchingconnect %w, %c0_ui8
    // CHECK: firrtl.matchingconnect %[[DATA]], %w_forced
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<8>
    // The `forced` register is anchored after the clk subfield of the ctrl
    // subfield, which is what clocks it.
    // CHECK: %[[CTRL_FV:.+]] = firrtl.subfield %[[CTRL]][forcedValue]
    // CHECK: %[[CTRL_CLK:.+]] = firrtl.subfield %[[CTRL]][clk]
    // CHECK: %forced = firrtl.reg %[[CTRL_CLK]] {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[CTRL_FV]], %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }

  // CHECK-LABEL: firrtl.module @ForceExportedThroughCast
  firrtl.module @ForceExportedThroughCast(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // CHECK: %c_probe_out = firrtl.instance c @ExportCastChild(out probe_out: !firrtl.bundle<data: uint<8>, ctrl flip: bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>>)
    // CHECK-NEXT: %[[CTRL:.+]] = firrtl.subfield %c_probe_out[ctrl]
    %p = firrtl.instance c @ExportCastChild(out probe_out: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.force %clock, %en, %p, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[CTRL_FA:.+]] = firrtl.subfield %[[CTRL]][forceActive]
    // CHECK: firrtl.matchingconnect %[[CTRL_FA]], %en : !firrtl.uint<1>
  }
}

// -----
// TEST 15: A module that only *re-exports* a child's probe (no local force)
// must not have its forwarding wire tied off: it is driven exactly once, from
// the inbound control port.  A tie-off here would be a second driver and would
// also drop the force from above.

// CHECK-LABEL: firrtl.circuit "ReExportNoLocalForce"
firrtl.circuit "ReExportNoLocalForce" {
  firrtl.module @RELeaf(out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
  }

  // CHECK-LABEL: firrtl.module @REMid
  // CHECK-SAME: out %p: !firrtl.bundle<data: uint<8>, ctrl flip: bundle
  firrtl.module @REMid(out %p: !firrtl.rwprobe<uint<8>>) {
    // The bundled probe port and the Leaf instance's bundled probe result are
    // both split into data/ctrl subfields; no forwarding wire is needed.
    // CHECK: %[[MID_CTRL:.+]] = firrtl.subfield %p[ctrl]
    // CHECK: %leaf_p = firrtl.instance leaf @RELeaf(out p: !firrtl.bundle<data: uint<8>, ctrl flip: bundle
    // CHECK: %[[LEAF_CTRL:.+]] = firrtl.subfield %leaf_p[ctrl]
    %lp = firrtl.instance leaf @RELeaf(out p: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.define %p, %lp : !firrtl.rwprobe<uint<8>>
    // The merge of %[[MID_CTRL]] fields onto %[[LEAF_CTRL]] fields (no-op
    // merge, no local force):
    // CHECK: %[[MID_FA:.+]] = firrtl.subfield %[[MID_CTRL]][forceActive]
    // CHECK: %[[LEAF_FA:.+]] = firrtl.subfield %[[LEAF_CTRL]][forceActive]
    // CHECK: firrtl.matchingconnect %[[LEAF_FA]], %[[MID_FA]]
  }

  // CHECK-LABEL: firrtl.module @ReExportNoLocalForce
  firrtl.module @ReExportNoLocalForce(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // CHECK: %mid_p = firrtl.instance mid @REMid(out p: !firrtl.bundle<data: uint<8>, ctrl flip: bundle
    // CHECK: %[[CTRL:.+]] = firrtl.subfield %mid_p[ctrl]
    %mp = firrtl.instance mid @REMid(out p: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.force %clock, %en, %mp, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[CTRL_FA:.+]] = firrtl.subfield %[[CTRL]][forceActive]
    // CHECK: firrtl.matchingconnect %[[CTRL_FA]], %en : !firrtl.uint<1>
  }
}

// -----
// TEST 16: Two forceable probes on the same instance, only one of which is
// forced.  The tie-off decision is per probe result: the unforced one gets the
// inactive default, the forced one is driven by the force reduction only.

// CHECK-LABEL: firrtl.module @MixedProbesOneInstance
firrtl.circuit "MixedProbesOneInstance" {
  firrtl.module @TwoProbes(out %pa: !firrtl.rwprobe<uint<8>>, out %pb: !firrtl.rwprobe<uint<4>>) {
    %a, %a_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %b, %b_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.define %pa, %a_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %pb, %b_ref : !firrtl.rwprobe<uint<4>>
  }

  firrtl.module @MixedProbesOneInstance(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // CHECK: %c_pa, %c_pb = firrtl.instance c @TwoProbes
    %pa, %pb = firrtl.instance c @TwoProbes(out pa: !firrtl.rwprobe<uint<8>>, out pb: !firrtl.rwprobe<uint<4>>)
    firrtl.ref.force %clock, %en, %pa, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[PA_CTRL:.+]] = firrtl.subfield %c_pa[ctrl]
    // CHECK: %[[PB_CTRL:.+]] = firrtl.subfield %c_pb[ctrl]

    // The forced probe `pa` (uint<8> payload) is driven from the force
    // reduction: the real clock, never a tie-off clock.
    // CHECK: %[[PA_FA:.+]] = firrtl.subfield %[[PA_CTRL]][forceActive]
    // CHECK: %[[PA_CLK:.+]] = firrtl.subfield %[[PA_CTRL]][clk]
    // CHECK: firrtl.matchingconnect %[[PA_FA]], %en : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %[[PA_CLK]], %clock : !firrtl.clock

    // The unforced probe `pb` (uint<4> payload) is the only one tied off, and
    // gets the constant-0 clock.
    // CHECK: %[[ZEROCLK:.+]] = firrtl.specialconstant 0 : !firrtl.clock
    // CHECK: %[[PB_CLK:.+]] = firrtl.subfield %[[PB_CTRL]][clk]
    // CHECK: firrtl.matchingconnect %[[PB_CLK]], %[[ZEROCLK]]
    // CHECK-NOT: firrtl.specialconstant
  }
}

// -----
// TEST 17: Release-only local target: a release with no force anywhere.  The
// reduction has no force value at all, so `invalid` is substituted for the
// `forcedValue` input (a null value used to crash here).  `forceActive` is a
// constant 0, so the state machine can never assert `forced`, the target keeps
// its original driver, and the whole override folds away downstream.

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

    // `forced` is only ever cleared (no force fires), so it holds its power-on
    // 0 forever and the substituted `invalid` is unreachable.
    // CHECK: %[[INV:.+]] = firrtl.invalidvalue : !firrtl.uint<8>
    // No sticky winner either: there is no force to latch.  The state register
    // is emitted after the invalid-value control input.
    // CHECK: %{{.+}} = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NOT: %forceWinner = firrtl.reg
    // CHECK: %[[FALSE:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[NEXT:.+]] = firrtl.mux(%rel, %[[FALSE]], %forced)
    // CHECK: firrtl.matchingconnect %forced, %[[NEXT]]
    // The target still follows its own driver whenever unforced.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[INV]], %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 18: Release-only of an instance probe.  The release is forwarded to the
// child's inbound control bundle with forceActive = 0 and forcedValue = invalid.

// CHECK-LABEL: firrtl.circuit "ReleaseOnlyInstance"
firrtl.circuit "ReleaseOnlyInstance" {
  // CHECK: firrtl.module @ROChild
  // CHECK-SAME: out %p: !firrtl.bundle<data: uint<8>, ctrl flip: bundle
  firrtl.module @ROChild(out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
  }

  // CHECK-LABEL: firrtl.module @ReleaseOnlyInstance
  firrtl.module @ReleaseOnlyInstance(in %clock: !firrtl.clock, in %rel: !firrtl.uint<1>) {
    // CHECK: %c_p = firrtl.instance c @ROChild
    // CHECK: %[[CTRL:.+]] = firrtl.subfield %c_p[ctrl]
    %p = firrtl.instance c @ROChild(out p: !firrtl.rwprobe<uint<8>>)
    firrtl.ref.release %clock, %rel, %p : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // The forwarding wire is driven by the release reduction (not tied off), so
    // the clock is the release's clock, not a constant-0 clock.
    // CHECK-NOT: firrtl.specialconstant
    // CHECK: %[[FALSE:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[INVALID:.+]] = firrtl.invalidvalue : !firrtl.uint<8>
    // CHECK: %[[FA:.+]] = firrtl.subfield %[[CTRL]][forceActive]
    // CHECK-NEXT: %[[RA:.+]] = firrtl.subfield %[[CTRL]][releaseActive]
    // CHECK-NEXT: %[[FV:.+]] = firrtl.subfield %[[CTRL]][forcedValue]
    // CHECK-NEXT: %[[CLK:.+]] = firrtl.subfield %[[CTRL]][clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[FA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[RA]], %rel
    // CHECK-NEXT: firrtl.matchingconnect %[[FV]], %[[INVALID]]
    // CHECK-NEXT: firrtl.matchingconnect %[[CLK]], %clock
  }
}

// -----
// TEST 19: Release-only of a target that is also exported.  The local release
// is merged with the inbound control: the release predicates are OR'd, and the
// (absent) local force leaves the inbound `forcedValue` as the only force data
// path.

// CHECK-LABEL: firrtl.circuit "ReleaseOnlyExported"
firrtl.circuit "ReleaseOnlyExported" {
  // CHECK: firrtl.module @ROExportChild
  // CHECK-SAME: out %p: !firrtl.bundle<data: uint<8>, ctrl flip: bundle
  firrtl.module @ROExportChild(in %clock: !firrtl.clock, in %rel: !firrtl.uint<1>, out %p: !firrtl.rwprobe<uint<8>>) {
    // The bundled probe port's control subfield %[[CTRL]] is read directly;
    // no separate inbound placeholder wire is needed.
    // CHECK: %[[CTRL:.+]] = firrtl.subfield %p[ctrl]
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    firrtl.matchingconnect %w, %c0 : !firrtl.uint<8>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
    firrtl.ref.release %clock, %rel, %w_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // The inbound control fields feed the merge; the local release is OR'd
    // into the inbound releaseActive from %[[CTRL]], and `forcedValue` comes
    // straight from %[[CTRL]][forcedValue] (no local force, so the mux folds).
    // CHECK: %[[IB_FA:.+]] = firrtl.subfield %[[CTRL]][forceActive]
    // CHECK: %[[IB_RA:.+]] = firrtl.subfield %[[CTRL]][releaseActive]
    // CHECK: %[[IB_FV:.+]] = firrtl.subfield %[[CTRL]][forcedValue]
    // CHECK: %[[OR:.+]] = firrtl.or %rel, %[[IB_RA]]
  }

  // CHECK-LABEL: firrtl.module @ReleaseOnlyExported
  firrtl.module @ReleaseOnlyExported(in %clock: !firrtl.clock, in %rel: !firrtl.uint<1>, in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    %c_clock, %c_rel, %c_p = firrtl.instance c @ROExportChild(in clock: !firrtl.clock, in rel: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<8>>)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_rel, %rel : !firrtl.uint<1>
    // The parent's force reaches the child through the new port.
    firrtl.ref.force %clock, %en, %c_p, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[CTRL:.+]] = firrtl.subfield %c_p[ctrl]
    // CHECK: firrtl.matchingconnect %[[FA:.+]], %en : !firrtl.uint<1>
  }
}

// -----
// TEST 20: Release-only of an instance probe through a same-type `ref.cast`.
// The cast is transparent, so the release reaches the child's control bundle.

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
    // CHECK: %c_p = firrtl.instance c @RCChild
    // CHECK: %[[CTRL:.+]] = firrtl.subfield %c_p[ctrl]
    %p = firrtl.instance c @RCChild(out p: !firrtl.rwprobe<uint<8>>)
    %cast = firrtl.ref.cast %p : (!firrtl.rwprobe<uint<8>>) -> !firrtl.rwprobe<uint<8>>
    firrtl.ref.release %clock, %rel, %cast : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>

    // No copy wire for the cast, no local state machine, and no inactive
    // tie-off of the forwarding wire.
    // CHECK-NOT: %forced = firrtl.reg
    // CHECK-NOT: firrtl.specialconstant
    // CHECK: %[[FALSE:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %[[FA:.+]] = firrtl.subfield %[[CTRL]][forceActive]
    // CHECK-NEXT: %[[RA:.+]] = firrtl.subfield %[[CTRL]][releaseActive]
    // CHECK-NEXT: %[[FV:.+]] = firrtl.subfield %[[CTRL]][forcedValue]
    // CHECK-NEXT: %[[CLK:.+]] = firrtl.subfield %[[CTRL]][clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[FA]], %[[FALSE]]
    // CHECK-NEXT: firrtl.matchingconnect %[[RA]], %rel
    // CHECK-NEXT: firrtl.matchingconnect %[[FV]], %{{.+}}
    // CHECK-NEXT: firrtl.matchingconnect %[[CLK]], %clock
  }
}

// -----
// TEST 21: A self-referential register next state (`r <= r + 1`) plus a clocked
// force.  The `+ 1` is a *read* of the target, so it goes through the override
// exactly like any other reader.  That matches the reference lowering: while
// forced, the procedural assignment is discarded but computed from the forced
// value, so releasing resumes from there rather than snapping back to the
// pre-force value.

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
    // The register keeps exactly one driver, its own next state.
    // CHECK: firrtl.matchingconnect %r, %[[NEXT]]
    // CHECK: firrtl.matchingconnect %o, %r_forced
    // The override itself reads the raw register (this is the only read that
    // does), so there is no combinational loop.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %v, %r)
    // CHECK: firrtl.matchingconnect %r_forced, %[[OVR]]
  }
}

// -----
// TEST 24: A forced target that nothing reads.  There is no read to override, so
// no observed wire and no mux are emitted at all -- and, crucially, the target
// keeps its own single driver.

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

    // No observed wire, no override mux, and the original driver is intact.
    // CHECK-NOT: firrtl.wire
    // CHECK: firrtl.matchingconnect %w, %[[C0]]
    // CHECK-NOT: firrtl.mux({{.*}}, %w)
  }
}

// -----
// TEST 25: Three clocked forces of one wire.  Only the force *event* is sampled:
// a one-hot latch records which force is currently in effect while every RHS
// stays live, so once all three predicates have dropped the target keeps
// tracking the *winning* force's value instead of collapsing back to the first
// one (which a plain last-wins mux over the current-cycle predicates would do).

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

    // No value register of the probed type: nothing is snapshotted.
    // CHECK-NOT: firrtl.reg {{.*}} !firrtl.uint<8>
    // forceActive is forceWins itself, the priority mux chain over the three
    // force predicates: it gates the latch so nothing moves on a cycle with
    // no force.
    // CHECK: %[[FW0:.+]] = firrtl.mux(%en2, %c1_ui1, %en1)
    // CHECK-NEXT: %[[FA:.+]] = firrtl.mux(%en3, %c1_ui1, %[[FW0]])
    // A later force masks an earlier one in the same cycle, so the second
    // force's select is gated on `not(en3)` while the third's is just `en3`.
    // CHECK-NEXT: %[[NOT_EN3:.+]] = firrtl.not %en3
    // CHECK-NEXT: %[[SEL2:.+]] = firrtl.and %en2, %[[NOT_EN3]]
    // One latch bit per force after the first; the first is the mux default.
    // CHECK: %forceWinner = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NEXT: %[[V2:.+]] = firrtl.mux(%forceWinner, %v2, %v1)
    // CHECK-NEXT: %forceWinner_0 = firrtl.reg %clock {initial = 0 : ui1, name = "forceWinner"} : !firrtl.clock, !firrtl.uint<1>
    // CHECK-NEXT: %[[V3:.+]] = firrtl.mux(%forceWinner_0, %v3, %[[V2]])
    // CHECK-NEXT: %[[WIN2:.+]] = firrtl.mux(%[[FA]], %[[SEL2]], %forceWinner)
    // CHECK-NEXT: firrtl.matchingconnect %forceWinner, %[[WIN2]]
    // CHECK-NEXT: %[[WIN3:.+]] = firrtl.mux(%[[FA]], %en3, %forceWinner_0)
    // CHECK-NEXT: firrtl.matchingconnect %forceWinner_0, %[[WIN3]]
    // The override drives the sticky *live* value.
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %[[V3]], %w)
    // CHECK: firrtl.matchingconnect %w_forced, %[[OVR]]
  }
}

// -----
// TEST 26: Force of a register clocked by a gated clock (`firrtl.int.clock_gate`).
// ProbesToSignals runs a gated-clock conversion first: the gate is eliminated
// from the clock path, the register is rebound to the ungated base clock, and
// the gate's enable becomes a data-path hold mux. The state machine sits
// entirely on the (now sole) base clock, and -- crucially -- the hold mux
// reads the *observed* (forced) value, so a forced register correctly stays
// forced while its gate is closed.

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

    // `forced` is on the base clock: only one clock domain remains.  It is
    // emitted after the hold logic at module end.
    // The gate becomes a hold mux over the *observed* value (so the register
    // holds the forced value, not the raw one, while gated), driving the
    // register's own next-state connect.
    // CHECK: %[[HOLD:.+]] = firrtl.mux(%gateEn, %d, %r_forced)
    // CHECK: firrtl.matchingconnect %r, %[[HOLD]]
    // CHECK: firrtl.matchingconnect %o, %r_forced
    // CHECK: %forced = firrtl.reg %clock {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %[[OVR:.+]] = firrtl.mux(%forced, %v, %r)
    // CHECK: firrtl.matchingconnect %r_forced, %[[OVR]]
  }
}

// -----
// TEST 27: Parent forces a child's exported register probe, where the
// register in the child is clocked by a gated clock local to the child. The
// clock gate is transparent to the pass: the child gets the same
// {data, ctrl} bundle port and control-merge shape as an ungated register
// (Test 14), and the parent's force -- on its own, unrelated clock -- reaches
// the child's control port exactly as it would for an ungated target.

// CHECK-LABEL: firrtl.circuit "ForceChildGatedClockRegister"
firrtl.circuit "ForceChildGatedClockRegister" {
  // The child's exported probe becomes a {data, ctrl} bundle port, same as
  // for an ungated register.
  // CHECK: firrtl.module @GatedChild(out %p: !firrtl.bundle<data: uint<8>, ctrl flip: bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>>
  firrtl.module @GatedChild(out %p: !firrtl.rwprobe<uint<8>>, in %clock: !firrtl.clock,
                            in %gateEn: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    // CHECK: %[[DATA:.+]] = firrtl.subfield %p[data]
    // CHECK: %[[CTRL:.+]] = firrtl.subfield %p[ctrl]
    %gated = firrtl.int.clock_gate %clock, %gateEn
    // The clock gate is converted away: the register is rebound to the base
    // clock and the gate becomes a data-path hold mux.
    // CHECK: %r = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    // CHECK-NOT: firrtl.int.clock_gate
    // CHECK: %r_forced = firrtl.wire : !firrtl.uint<8>
    %r, %r_ref = firrtl.reg %gated forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.ref.define %p, %r_ref : !firrtl.rwprobe<uint<8>>

    // Hold mux over the observed value; the exported data field is the
    // observed (forced) value too.
    // CHECK: %[[HOLD:.+]] = firrtl.mux(%gateEn, %d, %r_forced)
    // CHECK: firrtl.matchingconnect %r, %[[HOLD]]
    // CHECK: firrtl.matchingconnect %[[DATA]], %r_forced
    // The state machine (inbound control, same shape as Test 14).
    // CHECK: %[[CTRL_CLK:.+]] = firrtl.subfield %[[CTRL]][clk]
    // CHECK: %forced = firrtl.reg %[[CTRL_CLK]] {initial = 0 : ui1} : !firrtl.clock, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @ForceChildGatedClockRegister
  firrtl.module @ForceChildGatedClockRegister(in %clock: !firrtl.clock, in %gateEn: !firrtl.uint<1>,
                                              in %en: !firrtl.uint<1>, in %v: !firrtl.uint<8>) {
    // CHECK: %c_p, %c_clock, %c_gateEn, %c_d = firrtl.instance c @GatedChild(out p: !firrtl.bundle<data: uint<8>, ctrl flip: bundle
    // CHECK-NEXT: %[[CTRL:.+]] = firrtl.subfield %c_p[ctrl]
    %c_p, %c_clock, %c_gateEn, %c_d = firrtl.instance c @GatedChild(out p: !firrtl.rwprobe<uint<8>>, in clock: !firrtl.clock, in gateEn: !firrtl.uint<1>, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %c_gateEn, %gateEn : !firrtl.uint<1>

    // The parent forces with its own (ungated) clock, forwarded to the
    // child's control port -- unrelated to the child's local gated clock.
    firrtl.ref.force %clock, %en, %c_p, %v : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[CTRL_FA:.+]] = firrtl.subfield %[[CTRL]][forceActive]
    // CHECK-NEXT: %[[CTRL_RA:.+]] = firrtl.subfield %[[CTRL]][releaseActive]
    // CHECK-NEXT: %[[CTRL_FV:.+]] = firrtl.subfield %[[CTRL]][forcedValue]
    // CHECK-NEXT: %[[CTRL_CLK:.+]] = firrtl.subfield %[[CTRL]][clk]
    // CHECK: firrtl.matchingconnect %[[CTRL_FA]], %en
    // CHECK-NEXT: firrtl.matchingconnect %[[CTRL_RA]], %{{.+}}
    // CHECK-NEXT: firrtl.matchingconnect %[[CTRL_FV]], %v
    // CHECK-NEXT: firrtl.matchingconnect %[[CTRL_CLK]], %clock
  }
}
