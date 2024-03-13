// RUN: circt-opt -allow-unregistered-dialect --pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-comb-loops))' --split-input-file --verify-diagnostics %s

// Very simple tests involving rwprobe.

// When fixing these, move to check-comb-cycles.mlir and populate the diagnostics.

// This is effectively "XFAIL": Instead of marking XFAIL to lit, check no diagnostics are produced.
// Don't want this only appearing to pass due to diagnostic not matching, or only one not being caught.

// Check simple RWProbeOp + force loop.
firrtl.circuit "RWProbeOpLoop" {
  firrtl.module private @Foo(in %clock: !firrtl.clock sym @sym,
                             out %clockout: !firrtl.clock,
                             out %clockProbe_bore: !firrtl.rwprobe<clock>) {
    %0 = firrtl.ref.rwprobe <@Foo::@sym> : !firrtl.rwprobe<clock>
    firrtl.ref.define %clockProbe_bore, %0 : !firrtl.rwprobe<clock>
    firrtl.strictconnect %clockout, %clock : !firrtl.clock
  }

  // e.g., {{detected combinational cycle in a FIRRTL module, sample path: RWProbeOpLoop.}}
  firrtl.module @RWProbeOpLoop(in %clock: !firrtl.clock, out %clockProbe: !firrtl.rwprobe<clock>) {
    %foo_clock, %foo_clockout, %foo_clockProbe_bore = firrtl.instance foo @Foo(in clock: !firrtl.clock,
                                                                               out clockout: !firrtl.clock,
                                                                               out clockProbe_bore: !firrtl.rwprobe<clock>)
    firrtl.strictconnect %foo_clock, %clock : !firrtl.clock
    firrtl.ref.define %clockProbe, %foo_clockProbe_bore : !firrtl.rwprobe<clock>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.ref.force %clock, %c1_ui1, %clockProbe, %foo_clockout : !firrtl.clock, !firrtl.uint<1>, !firrtl.clock
  }
}

// -----

// Check simple RWProbeOp + read loop.

firrtl.circuit "RWProbeOpReadLoop" {
  firrtl.module private @Foo(in %clock: !firrtl.clock sym @sym,
                             out %clockProbe_bore: !firrtl.rwprobe<clock>) {
    %0 = firrtl.ref.rwprobe <@Foo::@sym> : !firrtl.rwprobe<clock>
    firrtl.ref.define %clockProbe_bore, %0 : !firrtl.rwprobe<clock>
  }

  // e.g., {{detected combinational cycle in a FIRRTL module, sample path: RWProbeOpReadLoop.}}
  firrtl.module @RWProbeOpReadLoop() {
    %foo_clock, %foo_clockProbe_bore = firrtl.instance foo @Foo(in clock: !firrtl.clock,
                                                                out clockProbe_bore: !firrtl.rwprobe<clock>)
    %read = firrtl.ref.resolve %foo_clockProbe_bore : !firrtl.rwprobe<clock>
    firrtl.strictconnect %foo_clock, %read : !firrtl.clock
  }
}
