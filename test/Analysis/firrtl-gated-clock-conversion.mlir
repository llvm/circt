// RUN: circt-opt %s -test-firrtl-gated-clock-conversion -split-input-file | FileCheck %s

// Cascaded gates in one module: the enables are ANDed.
// CHECK-LABEL: firrtl.module @Cascaded
firrtl.circuit "Cascaded" {
  firrtl.module @Cascaded(in %clk: !firrtl.clock, in %en1: !firrtl.uint<1>,
                          in %en2: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    // CHECK: %[[AND:.+]] = firrtl.and %en1, %en2
    // CHECK: %[[R:.+]] = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %[[MUX:.+]] = firrtl.mux(%[[AND]], %d, %[[R]])
    // CHECK: firrtl.matchingconnect %[[R]], %[[MUX]]
    %g1 = firrtl.int.clock_gate %clk, %en1
    %g2 = firrtl.int.clock_gate %g1, %en2
    %r = firrtl.reg %g2 : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

// A wire alias is looked through; no extra wire is needed when the base clock
// dominates the register.
// CHECK-LABEL: firrtl.module @WireAlias
firrtl.circuit "WireAlias" {
  firrtl.module @WireAlias(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                           in %d: !firrtl.uint<8>) {
    // CHECK-NOT: firrtl.wire
    // CHECK: firrtl.int.clock_gate %clk, %en
    // CHECK: %[[W:.+]] = firrtl.wire : !firrtl.clock
    // CHECK: %[[R:.+]] = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %[[MUX:.+]] = firrtl.mux(%en, %d, %[[R]])
    // CHECK: firrtl.matchingconnect %[[R]], %[[MUX]]
    %g = firrtl.int.clock_gate %clk, %en
    %w = firrtl.wire : !firrtl.clock
    firrtl.matchingconnect %w, %g : !firrtl.clock
    %r = firrtl.reg %w : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

// An ungated register is left alone.
// CHECK-LABEL: firrtl.module @NoGate
firrtl.circuit "NoGate" {
  firrtl.module @NoGate(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    // CHECK: %[[R:.+]] = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    // CHECK-NEXT: firrtl.matchingconnect %[[R]], %d
    // CHECK-NOT: firrtl.mux
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

// Only the root on the gated clock is rewritten.
// CHECK-LABEL: firrtl.module @ForceOnBase
firrtl.circuit "ForceOnBase" {
  firrtl.module @ForceOnBase(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                             in %cond: !firrtl.uint<1>, in %val: !firrtl.uint<8>,
                             in %d: !firrtl.uint<8>) {
    // CHECK: %[[R:.+]], %[[RREF:.+]] = firrtl.reg %clk forceable
    // CHECK: firrtl.mux(%en, %d, %[[R]])
    // CHECK: firrtl.ref.force %clk, %cond, %[[RREF]], %val
    // CHECK-NOT: firrtl.and
    %g = firrtl.int.clock_gate %clk, %en
    %r, %r_ref = firrtl.reg %g forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.ref.force %clk, %cond, %r_ref, %val : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

// The test enable is ORed in. The enable is defined after the force and
// release, so it reaches them through a wire.
// CHECK-LABEL: firrtl.module @EnableAfterRoot
firrtl.circuit "EnableAfterRoot" {
  firrtl.module @EnableAfterRoot(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                                 in %te: !firrtl.uint<1>, in %p: !firrtl.uint<1>,
                                 in %d: !firrtl.uint<8>) {
    // CHECK: %[[EN_WIRE:.+]] = firrtl.wire : !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.clock
    %x, %x_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %x, %d : !firrtl.uint<8>
    // CHECK: %[[P1:.+]] = firrtl.and %p, %[[EN_WIRE]]
    // CHECK: firrtl.ref.force %clk, %[[P1]], %x_ref, %d
    firrtl.ref.force %w, %p, %x_ref, %d : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    // CHECK: %[[P2:.+]] = firrtl.and %p, %[[EN_WIRE]]
    // CHECK: firrtl.ref.release %clk, %[[P2]], %x_ref
    firrtl.ref.release %w, %p, %x_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
    // CHECK: %[[OR:.+]] = firrtl.or %en, %te
    // CHECK: %[[AND:.+]] = firrtl.and %[[OR]], %en
    // CHECK: firrtl.matchingconnect %[[EN_WIRE]], %[[AND]]
    %g = firrtl.int.clock_gate %clk, %en, %te
    %g2 = firrtl.int.clock_gate %g, %en
    firrtl.matchingconnect %w, %g2 : !firrtl.clock
  }
}

// -----

// The base clock, an extmodule output, is defined after the register, so it
// reaches it through a wire.
// CHECK-LABEL: firrtl.module @BaseAfterRoot
firrtl.circuit "BaseAfterRoot" {
  firrtl.extmodule @Ext(out clk: !firrtl.clock)
  firrtl.module @BaseAfterRoot(in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    // CHECK: %[[BASE_WIRE:.+]] = firrtl.wire : !firrtl.clock
    %w = firrtl.wire : !firrtl.clock
    %g = firrtl.int.clock_gate %w, %en
    // CHECK: %r = firrtl.reg %[[BASE_WIRE]]
    %r = firrtl.reg %g : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    // CHECK: %[[BASE:.+]] = firrtl.instance x @Ext
    %x = firrtl.instance x @Ext(out clk: !firrtl.clock)
    // CHECK: firrtl.matchingconnect %[[BASE_WIRE]], %[[BASE]] : !firrtl.clock
    firrtl.matchingconnect %w, %x : !firrtl.clock
  }
}

// -----

// `asClock(u)` is a base clock of its own, not an alias of `u`.
// CHECK-LABEL: firrtl.circuit "CastOfInt"
firrtl.circuit "CastOfInt" {
  // CHECK: firrtl.module private @Child
  firrtl.module private @Child(in %u: !firrtl.uint<1>, in %en: !firrtl.uint<1>,
                               out %o: !firrtl.clock) {
    // CHECK: %[[CAST:.+]] = firrtl.asClock %u
    %c = firrtl.asClock %u : (!firrtl.uint<1>) -> !firrtl.clock
    %n = firrtl.node %c : !firrtl.clock
    %g = firrtl.int.clock_gate %n, %en
    // CHECK: firrtl.matchingconnect %_gatedClock_baseClock_o, %[[CAST]] : !firrtl.clock
    firrtl.matchingconnect %o, %g : !firrtl.clock
  }
  firrtl.module @CastOfInt(in %u: !firrtl.uint<1>, in %en: !firrtl.uint<1>,
                           in %d: !firrtl.uint<8>) {
    %c_u, %c_en, %c_o = firrtl.instance c @Child(in u: !firrtl.uint<1>,
      in en: !firrtl.uint<1>, out o: !firrtl.clock)
    firrtl.matchingconnect %c_u, %u : !firrtl.uint<1>
    firrtl.matchingconnect %c_en, %en : !firrtl.uint<1>
    // CHECK: firrtl.reg %c__gatedClock_baseClock_o
    %r = firrtl.reg %c_o : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

// Instances nested in layerblocks.
// CHECK-LABEL: firrtl.module @DeepChild
// CHECK-SAME:    in %[[DBASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk]]: !firrtl.clock
// CHECK-SAME:    in %[[DEN:[A-Za-z0-9_]*_gatedClock_enable_clk]]: !firrtl.uint<1>
// CHECK:         %[[R:.+]] = firrtl.reg %[[DBASE]] : !firrtl.clock, !firrtl.uint<8>
// CHECK:         %[[MUX:.+]] = firrtl.mux(%[[DEN]], %data, %[[R]])
// CHECK:         firrtl.matchingconnect %[[R]], %[[MUX]]

// CHECK-LABEL: firrtl.module @DeeplyNested
firrtl.circuit "DeeplyNested" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {}
  }

  firrtl.module @DeepChild(in %clk: !firrtl.clock, in %data: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %data : !firrtl.uint<8>
  }

  firrtl.module @DeeplyNested(in %clk: !firrtl.clock,
                              in %en: !firrtl.uint<1>,
                              in %selector: !firrtl.enum<A: uint<8>, B: uint<8>>,
                              in %cond: !firrtl.uint<1>,
                              in %d: !firrtl.uint<8>) {
    // CHECK: firrtl.int.clock_gate %clk, %en
    %g = firrtl.int.clock_gate %clk, %en

    // CHECK: firrtl.layerblock @A
    // CHECK: firrtl.layerblock @A::@B
    // CHECK: %{{.+}}, %{{.+}}, %[[INST_BASE:.+]], %[[INST_EN:.+]] = firrtl.instance deep_child @DeepChild
    // CHECK-SAME: in {{[A-Za-z0-9_]*_gatedClock_baseClock_clk}}: !firrtl.clock
    // CHECK-SAME: in {{[A-Za-z0-9_]*_gatedClock_enable_clk}}: !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %[[INST_BASE]], %clk
    // CHECK: firrtl.matchingconnect %[[INST_EN]], %en
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
        %child_clk, %child_data = firrtl.instance deep_child @DeepChild(in clk: !firrtl.clock, in data: !firrtl.uint<8>)
        firrtl.matchingconnect %child_clk, %g : !firrtl.clock
        firrtl.matchingconnect %child_data, %d : !firrtl.uint<8>
      }
    }
  }
}

// -----

// Each port gets a pair if any caller gates it; ungated callers drive a
// constant 1 enable.
// CHECK-LABEL: firrtl.module @FlexibleChild
// CHECK-SAME:    in %clk1: !firrtl.clock
// CHECK-SAME:    in %clk2: !firrtl.clock
// CHECK-SAME:    in %data: !firrtl.uint<8>
// CHECK-SAME:    in %[[CLK1_BASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk1]]: !firrtl.clock
// CHECK-SAME:    in %[[CLK1_EN:[A-Za-z0-9_]*_gatedClock_enable_clk1]]: !firrtl.uint<1>
// CHECK-SAME:    in %[[CLK2_BASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk2]]: !firrtl.clock
// CHECK-SAME:    in %[[CLK2_EN:[A-Za-z0-9_]*_gatedClock_enable_clk2]]: !firrtl.uint<1>
// CHECK:         %[[R1:.+]] = firrtl.reg %[[CLK1_BASE]] : !firrtl.clock, !firrtl.uint<8>
// CHECK:         %[[R2:.+]] = firrtl.reg %[[CLK2_BASE]] : !firrtl.clock, !firrtl.uint<8>
// CHECK:         %[[MUX1:.+]] = firrtl.mux(%[[CLK1_EN]], %data, %[[R1]])
// CHECK:         firrtl.matchingconnect %[[R1]], %[[MUX1]]
// CHECK:         %[[MUX2:.+]] = firrtl.mux(%[[CLK2_EN]], %data, %[[R2]])
// CHECK:         firrtl.matchingconnect %[[R2]], %[[MUX2]]

// CHECK-LABEL: firrtl.module @DifferentGating
firrtl.circuit "DifferentGating" {
  firrtl.module @FlexibleChild(in %clk1: !firrtl.clock,
                               in %clk2: !firrtl.clock,
                               in %data: !firrtl.uint<8>) {
    %r1 = firrtl.reg %clk1 : !firrtl.clock, !firrtl.uint<8>
    %r2 = firrtl.reg %clk2 : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r1, %data : !firrtl.uint<8>
    firrtl.matchingconnect %r2, %data : !firrtl.uint<8>
  }

  firrtl.module @DifferentGating(in %base_clk: !firrtl.clock,
                                 in %en1: !firrtl.uint<1>,
                                 in %en2: !firrtl.uint<1>,
                                 in %d: !firrtl.uint<8>) {
    // CHECK: %[[C1:.+]] = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: firrtl.int.clock_gate %base_clk, %en1
    %g1 = firrtl.int.clock_gate %base_clk, %en1
    // CHECK: firrtl.int.clock_gate %base_clk, %en2
    %g2 = firrtl.int.clock_gate %base_clk, %en2

    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[I1_CLK1_BASE:.+]], %[[I1_CLK1_EN:.+]], %[[I1_CLK2_BASE:.+]], %[[I1_CLK2_EN:.+]] = firrtl.instance inst1 @FlexibleChild
    %inst1_clk1, %inst1_clk2, %inst1_data = firrtl.instance inst1 @FlexibleChild(
      in clk1: !firrtl.clock,
      in clk2: !firrtl.clock,
      in data: !firrtl.uint<8>)
    firrtl.matchingconnect %inst1_clk1, %g1 : !firrtl.clock
    firrtl.matchingconnect %inst1_clk2, %base_clk : !firrtl.clock
    firrtl.matchingconnect %inst1_data, %d : !firrtl.uint<8>

    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[I2_CLK1_BASE:.+]], %[[I2_CLK1_EN:.+]], %[[I2_CLK2_BASE:.+]], %[[I2_CLK2_EN:.+]] = firrtl.instance inst2 @FlexibleChild
    %inst2_clk1, %inst2_clk2, %inst2_data = firrtl.instance inst2 @FlexibleChild(
      in clk1: !firrtl.clock,
      in clk2: !firrtl.clock,
      in data: !firrtl.uint<8>)
    firrtl.matchingconnect %inst2_clk1, %base_clk : !firrtl.clock
    firrtl.matchingconnect %inst2_clk2, %g2 : !firrtl.clock
    firrtl.matchingconnect %inst2_data, %d : !firrtl.uint<8>
    // CHECK-DAG: firrtl.matchingconnect %[[I1_CLK2_BASE]], %base_clk
    // CHECK-DAG: firrtl.matchingconnect %[[I1_CLK2_EN]], %[[C1]]
    // CHECK-DAG: firrtl.matchingconnect %[[I2_CLK2_BASE]], %base_clk
    // CHECK-DAG: firrtl.matchingconnect %[[I2_CLK2_EN]], %en2
    // CHECK-DAG: firrtl.matchingconnect %[[I2_CLK1_BASE]], %base_clk
    // CHECK-DAG: firrtl.matchingconnect %[[I2_CLK1_EN]], %[[C1]]
    // CHECK-DAG: firrtl.matchingconnect %[[I1_CLK1_BASE]], %base_clk
    // CHECK-DAG: firrtl.matchingconnect %[[I1_CLK1_EN]], %en1
  }
}

// -----

// Pairs are added per port: `clk_in1` and the `clk_out1` forwarding it get
// pairs, `clk_in2` and `clk_out2` do not.
// CHECK-LABEL: firrtl.module @MultiIOClock
// CHECK-SAME:    in %clk_in1: !firrtl.clock
// CHECK-SAME:    in %clk_in2: !firrtl.clock
// CHECK-SAME:    out %clk_out1: !firrtl.clock
// CHECK-SAME:    out %clk_out2: !firrtl.clock
// CHECK-SAME:    in %data: !firrtl.uint<16>
// CHECK-SAME:    in %[[BASE_IN1:[A-Za-z0-9_]*_gatedClock_baseClock_clk_in1]]: !firrtl.clock
// CHECK-SAME:    in %[[EN_IN1:[A-Za-z0-9_]*_gatedClock_enable_clk_in1]]: !firrtl.uint<1>
// CHECK-SAME:    out %[[BASE_OUT1:[A-Za-z0-9_]*_gatedClock_baseClock_clk_out1]]: !firrtl.clock
// CHECK-SAME:    out %[[EN_OUT1:[A-Za-z0-9_]*_gatedClock_enable_clk_out1]]: !firrtl.uint<1>
// CHECK-NOT:     _gatedClock_baseClock_clk_in2
// CHECK-NOT:     _gatedClock_baseClock_clk_out2

firrtl.circuit "MultiIOClockMultiInst" {
  firrtl.module @MultiIOClock(in %clk_in1: !firrtl.clock, in %clk_in2: !firrtl.clock,
                              out %clk_out1: !firrtl.clock, out %clk_out2: !firrtl.clock,
                              in %data: !firrtl.uint<16>) {
    // CHECK: %[[R1:.+]] = firrtl.reg %[[BASE_IN1]] : !firrtl.clock, !firrtl.uint<16>
    // CHECK: %[[MUX1:.+]] = firrtl.mux(%[[EN_IN1]], %data, %[[R1]])
    // CHECK: firrtl.matchingconnect %[[R1]], %[[MUX1]]
    %r1 = firrtl.reg %clk_in1 : !firrtl.clock, !firrtl.uint<16>
    firrtl.matchingconnect %r1, %data : !firrtl.uint<16>

    // CHECK: %[[R2:.+]] = firrtl.reg %clk_in2 : !firrtl.clock, !firrtl.uint<16>
    // CHECK: firrtl.matchingconnect %[[R2]], %data
    %r2 = firrtl.reg %clk_in2 : !firrtl.clock, !firrtl.uint<16>
    firrtl.matchingconnect %r2, %data : !firrtl.uint<16>

    // CHECK: firrtl.matchingconnect %clk_out1, %clk_in1
    // CHECK: firrtl.matchingconnect %clk_out2, %clk_in2
    firrtl.matchingconnect %clk_out1, %clk_in1 : !firrtl.clock
    firrtl.matchingconnect %clk_out2, %clk_in2 : !firrtl.clock
    // CHECK: firrtl.matchingconnect %[[BASE_OUT1]], %[[BASE_IN1]]
    // CHECK: firrtl.matchingconnect %[[EN_OUT1]], %[[EN_IN1]]
  }

  // CHECK-LABEL: firrtl.module @MultiIOClockMultiInst
  firrtl.module @MultiIOClockMultiInst(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<16>) {
    // CHECK: %[[C1:.+]] = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: %[[G:.+]] = firrtl.int.clock_gate %clk, %en
    %g = firrtl.int.clock_gate %clk, %en

    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[I1_BASE_IN:.+]], %[[I1_EN_IN:.+]], %[[I1_BASE_OUT:.+]], %[[I1_EN_OUT:.+]] = firrtl.instance inst1 @MultiIOClock
    %i1_in1, %i1_in2, %i1_out1, %i1_out2, %i1_d = firrtl.instance inst1 @MultiIOClock(
      in clk_in1: !firrtl.clock, in clk_in2: !firrtl.clock,
      out clk_out1: !firrtl.clock, out clk_out2: !firrtl.clock,
      in data: !firrtl.uint<16>)
    firrtl.matchingconnect %i1_in1, %g : !firrtl.clock
    firrtl.matchingconnect %i1_in2, %clk : !firrtl.clock
    firrtl.matchingconnect %i1_d, %d : !firrtl.uint<16>

    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[I2_BASE_IN:.+]], %[[I2_EN_IN:.+]], %{{.+}}, %{{.+}} = firrtl.instance inst2 @MultiIOClock
    %i2_in1, %i2_in2, %i2_out1, %i2_out2, %i2_d = firrtl.instance inst2 @MultiIOClock(
      in clk_in1: !firrtl.clock, in clk_in2: !firrtl.clock,
      out clk_out1: !firrtl.clock, out clk_out2: !firrtl.clock,
      in data: !firrtl.uint<16>)
    firrtl.matchingconnect %i2_in1, %clk : !firrtl.clock
    firrtl.matchingconnect %i2_in2, %clk : !firrtl.clock
    firrtl.matchingconnect %i2_d, %d : !firrtl.uint<16>

    // CHECK: %[[TR1:.+]] = firrtl.reg %[[I1_BASE_OUT]] : !firrtl.clock, !firrtl.uint<16>
    // CHECK: %[[TMUX:.+]] = firrtl.mux(%[[I1_EN_OUT]], %d, %[[TR1]])
    // CHECK: firrtl.matchingconnect %[[TR1]], %[[TMUX]]
    %r1 = firrtl.reg %i1_out1 : !firrtl.clock, !firrtl.uint<16>
    firrtl.matchingconnect %r1, %d : !firrtl.uint<16>

    // CHECK-DAG: firrtl.matchingconnect %[[I2_BASE_IN]], %clk
    // CHECK-DAG: firrtl.matchingconnect %[[I2_EN_IN]], %[[C1]]
    // CHECK-DAG: firrtl.matchingconnect %[[I1_BASE_IN]], %clk
    // CHECK-DAG: firrtl.matchingconnect %[[I1_EN_IN]], %en
  }
}

// -----

// A cascade of two instances of one module. Pairs are relative to the module's
// own ports, so the cyclic clock graph needs no special handling.
// CHECK-LABEL: firrtl.module @ClockGen
// CHECK-SAME:    in %[[BASE_IN:[A-Za-z0-9_]*_gatedClock_baseClock_clk_in]]: !firrtl.clock
// CHECK-SAME:    in %[[EN_IN:[A-Za-z0-9_]*_gatedClock_enable_clk_in]]: !firrtl.uint<1>
// CHECK-SAME:    out %[[BASE_OUT:[A-Za-z0-9_]*_gatedClock_baseClock_clk_out]]: !firrtl.clock
// CHECK-SAME:    out %[[EN_OUT:[A-Za-z0-9_]*_gatedClock_enable_clk_out]]: !firrtl.uint<1>
firrtl.circuit "Top" {
  firrtl.module @ClockGen(in %clk_in: !firrtl.clock, in %en: !firrtl.uint<1>, out %clk_out: !firrtl.clock) {
    // CHECK: %[[AND:.+]] = firrtl.and %[[EN_IN]], %en
    // CHECK: %[[G:.+]] = firrtl.int.clock_gate %clk_in, %en
    %g = firrtl.int.clock_gate %clk_in, %en
    // CHECK: %[[RR:.+]] = firrtl.reg %[[BASE_IN]] : !firrtl.clock, !firrtl.uint<4>
    // CHECK: %[[C3:.+]] = firrtl.constant 3 : !firrtl.uint<4>
    // CHECK: %[[BMUX:.+]] = firrtl.mux(%[[EN_IN]], %[[C3]], %[[RR]])
    // CHECK: firrtl.matchingconnect %[[RR]], %[[BMUX]]
    %rr = firrtl.reg %clk_in : !firrtl.clock, !firrtl.uint<4>
    %c0 = firrtl.constant 3 : !firrtl.uint<4>
    firrtl.matchingconnect %rr, %c0 : !firrtl.uint<4>
    // CHECK: firrtl.matchingconnect %clk_out, %[[G]]
    // CHECK: firrtl.matchingconnect %[[BASE_OUT]], %[[BASE_IN]]
    // CHECK: firrtl.matchingconnect %[[EN_OUT]], %[[AND]]
    firrtl.matchingconnect %clk_out, %g : !firrtl.clock
  }

  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %clk: !firrtl.clock, in %en1: !firrtl.uint<1>, in %en2: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    // CHECK: %[[C1:.+]] = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[G1_BASE_IN:.+]], %[[G1_EN_IN:.+]], %[[G1_BASE_OUT:.+]], %[[G1_EN_OUT:.+]] = firrtl.instance gen1 @ClockGen
    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[G2_BASE_IN:.+]], %[[G2_EN_IN:.+]], %[[G2_BASE_OUT:.+]], %[[G2_EN_OUT:.+]] = firrtl.instance gen2 @ClockGen
    %g1_in, %g1_en, %g1_out = firrtl.instance gen1 @ClockGen(in clk_in: !firrtl.clock, in en: !firrtl.uint<1>, out clk_out: !firrtl.clock)
    %g2_in, %g2_en, %g2_out = firrtl.instance gen2 @ClockGen(in clk_in: !firrtl.clock, in en: !firrtl.uint<1>, out clk_out: !firrtl.clock)
    firrtl.matchingconnect %g1_in, %clk : !firrtl.clock
    firrtl.matchingconnect %g1_en, %en1 : !firrtl.uint<1>
    firrtl.matchingconnect %g2_in, %g1_out : !firrtl.clock
    firrtl.matchingconnect %g2_en, %en2 : !firrtl.uint<1>

    // CHECK: %[[R:.+]] = firrtl.reg %[[G2_BASE_OUT]] : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %[[MUX:.+]] = firrtl.mux(%[[G2_EN_OUT]], %d, %[[R]])
    %r = firrtl.reg %g2_out : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>

    // CHECK-DAG: firrtl.matchingconnect %[[G1_BASE_IN]], %clk
    // CHECK-DAG: firrtl.matchingconnect %[[G1_EN_IN]], %[[C1]]
    // CHECK-DAG: firrtl.matchingconnect %[[G2_BASE_IN]], %[[G1_BASE_OUT]]
    // CHECK-DAG: firrtl.matchingconnect %[[G2_EN_IN]], %[[G1_EN_OUT]]
  }
}

// -----

// A module without gates forwards its child's pairs in both directions.
// CHECK-LABEL: firrtl.module @ClockGen
// CHECK-SAME:    in %[[CG_BI:[A-Za-z0-9_]*_gatedClock_baseClock_clk_in]]: !firrtl.clock
// CHECK-SAME:    in %[[CG_EI:[A-Za-z0-9_]*_gatedClock_enable_clk_in]]: !firrtl.uint<1>
// CHECK-SAME:    out %[[CG_BO:[A-Za-z0-9_]*_gatedClock_baseClock_clk_out]]: !firrtl.clock
// CHECK-SAME:    out %[[CG_EO:[A-Za-z0-9_]*_gatedClock_enable_clk_out]]: !firrtl.uint<1>
firrtl.circuit "Top" {
  firrtl.module @ClockGen(in %clk_in: !firrtl.clock, in %en: !firrtl.uint<1>, out %clk_out: !firrtl.clock) {
    // CHECK: %[[AND:.+]] = firrtl.and %[[CG_EI]], %en
    // CHECK: firrtl.matchingconnect %[[CG_BO]], %[[CG_BI]]
    // CHECK: firrtl.matchingconnect %[[CG_EO]], %[[AND]]
    %g = firrtl.int.clock_gate %clk_in, %en
    firrtl.matchingconnect %clk_out, %g : !firrtl.clock
  }

  // CHECK-LABEL: firrtl.module @Mid
  // CHECK-SAME:    in %[[M_BI:[A-Za-z0-9_]*_gatedClock_baseClock_clk_in]]: !firrtl.clock
  // CHECK-SAME:    in %[[M_EI:[A-Za-z0-9_]*_gatedClock_enable_clk_in]]: !firrtl.uint<1>
  // CHECK-SAME:    out %[[M_BO:[A-Za-z0-9_]*_gatedClock_baseClock_clk_out]]: !firrtl.clock
  // CHECK-SAME:    out %[[M_EO:[A-Za-z0-9_]*_gatedClock_enable_clk_out]]: !firrtl.uint<1>
  firrtl.module @Mid(in %clk_in: !firrtl.clock, in %en: !firrtl.uint<1>, out %clk_out: !firrtl.clock) {
    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[V_BI:.+]], %[[V_EI:.+]], %[[V_BO:.+]], %[[V_EO:.+]] = firrtl.instance v @ClockGen
    %v_i, %v_e, %v_o = firrtl.instance v @ClockGen(in clk_in: !firrtl.clock, in en: !firrtl.uint<1>, out clk_out: !firrtl.clock)
    firrtl.matchingconnect %v_i, %clk_in : !firrtl.clock
    firrtl.matchingconnect %v_e, %en : !firrtl.uint<1>
    firrtl.matchingconnect %clk_out, %v_o : !firrtl.clock
    // CHECK-DAG: firrtl.matchingconnect %[[M_BO]], %[[V_BO]]
    // CHECK-DAG: firrtl.matchingconnect %[[M_EO]], %[[V_EO]]
    // CHECK-DAG: firrtl.matchingconnect %[[V_BI]], %[[M_BI]]
    // CHECK-DAG: firrtl.matchingconnect %[[V_EI]], %[[M_EI]]
  }

  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %clk: !firrtl.clock, in %e1: !firrtl.uint<1>, in %e2: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    // CHECK: %[[C1:.+]] = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[M1_BI:.+]], %[[M1_EI:.+]], %[[M1_BO:.+]], %[[M1_EO:.+]] = firrtl.instance m1 @Mid
    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[M2_BI:.+]], %[[M2_EI:.+]], %[[M2_BO:.+]], %[[M2_EO:.+]] = firrtl.instance m2 @Mid
    %m1_i, %m1_e, %m1_o = firrtl.instance m1 @Mid(in clk_in: !firrtl.clock, in en: !firrtl.uint<1>, out clk_out: !firrtl.clock)
    %m2_i, %m2_e, %m2_o = firrtl.instance m2 @Mid(in clk_in: !firrtl.clock, in en: !firrtl.uint<1>, out clk_out: !firrtl.clock)
    firrtl.matchingconnect %m1_i, %clk : !firrtl.clock
    firrtl.matchingconnect %m1_e, %e1 : !firrtl.uint<1>
    firrtl.matchingconnect %m2_i, %m1_o : !firrtl.clock
    firrtl.matchingconnect %m2_e, %e2 : !firrtl.uint<1>

    // CHECK: %[[R:.+]] = firrtl.reg %[[M2_BO]] : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %[[MUX:.+]] = firrtl.mux(%[[M2_EO]], %d, %[[R]])
    %r = firrtl.reg %m2_o : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>

    // CHECK-DAG: firrtl.matchingconnect %[[M1_BI]], %clk
    // CHECK-DAG: firrtl.matchingconnect %[[M1_EI]], %[[C1]]
    // CHECK-DAG: firrtl.matchingconnect %[[M2_BI]], %[[M1_BO]]
    // CHECK-DAG: firrtl.matchingconnect %[[M2_EI]], %[[M1_EO]]
  }
}

// -----

// A cascade across an output port into a gate of the caller, whose register
// is clocked through a wire driven later.
firrtl.circuit "OutputCascadeAfterRoot" {
  // CHECK-LABEL: firrtl.module @Gen
  // CHECK-SAME:    out %[[BASE_OUT:_gatedClock_baseClock_o]]: !firrtl.clock
  // CHECK-SAME:    out %[[EN_OUT:_gatedClock_enable_o]]: !firrtl.uint<1>
  // CHECK:         firrtl.matchingconnect %[[BASE_OUT]], %clk
  // CHECK:         firrtl.matchingconnect %[[EN_OUT]], %en
  firrtl.module @Gen(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, out %o: !firrtl.clock) {
    %g = firrtl.int.clock_gate %clk, %en
    %n = firrtl.node %g : !firrtl.clock
    firrtl.matchingconnect %o, %n : !firrtl.clock
  }
  // CHECK-LABEL: firrtl.module @OutputCascadeAfterRoot
  firrtl.module @OutputCascadeAfterRoot(in %clk: !firrtl.clock, in %e1: !firrtl.uint<1>,
                                        in %e2: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    // CHECK: %[[EN_WIRE:.+]] = firrtl.wire : !firrtl.uint<1>
    // CHECK: %[[CLK_WIRE:.+]] = firrtl.wire : !firrtl.clock
    %w = firrtl.wire : !firrtl.clock
    // CHECK: %r = firrtl.regreset %[[CLK_WIRE]], %e1, %d
    // CHECK: %[[MUX:.+]] = firrtl.mux(%[[EN_WIRE]], %d, %r)
    // CHECK: firrtl.matchingconnect %r, %[[MUX]]
    %r = firrtl.regreset %w, %e1, %d : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[G_BASE:.+]], %[[G_EN:.+]] = firrtl.instance gen @Gen
    // CHECK: firrtl.matchingconnect %[[CLK_WIRE]], %[[G_BASE]]
    // CHECK: %[[AND:.+]] = firrtl.and %[[G_EN]], %e2
    // CHECK: firrtl.matchingconnect %[[EN_WIRE]], %[[AND]]
    %c, %e, %o = firrtl.instance gen @Gen(in clk: !firrtl.clock, in en: !firrtl.uint<1>, out o: !firrtl.clock)
    firrtl.matchingconnect %c, %clk : !firrtl.clock
    firrtl.matchingconnect %e, %e1 : !firrtl.uint<1>
    %g = firrtl.int.clock_gate %o, %e2
    firrtl.matchingconnect %w, %g : !firrtl.clock
  }
}

// -----

// `c1` is re-created for `Child`'s new ports, so the register must be clocked
// by the new instance's ungated output.
firrtl.circuit "UngatedOutputWithGatedInput" {
  // CHECK-LABEL: firrtl.module @GrandChild
  // CHECK-SAME:    in %[[GC_BASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk]]: !firrtl.clock
  // CHECK-SAME:    in %[[GC_EN:[A-Za-z0-9_]*_gatedClock_enable_clk]]: !firrtl.uint<1>
  // CHECK:         %[[GC_REG:.+]] = firrtl.reg %[[GC_BASE]]
  // CHECK:         %[[GC_MUX:.+]] = firrtl.mux(%[[GC_EN]], %data, %[[GC_REG]])
  // CHECK:         firrtl.matchingconnect %[[GC_REG]], %[[GC_MUX]]
  firrtl.module @GrandChild(in %clk: !firrtl.clock, in %data: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %data : !firrtl.uint<8>
  }

  // CHECK-LABEL: firrtl.module @Child
  // CHECK-SAME:    in %[[CHILD_BASE:[A-Za-z0-9_]*_gatedClock_baseClock_gated_in]]: !firrtl.clock
  // CHECK-SAME:    in %[[CHILD_EN:[A-Za-z0-9_]*_gatedClock_enable_gated_in]]: !firrtl.uint<1>
  // CHECK:         firrtl.matchingconnect %clk_out, %clk_in
  // CHECK:         firrtl.matchingconnect %[[GC_BASE_IN:.+]], %[[CHILD_BASE]]
  // CHECK:         firrtl.matchingconnect %[[GC_EN_IN:.+]], %[[CHILD_EN]]
  firrtl.module @Child(in %gated_in: !firrtl.clock, in %clk_in: !firrtl.clock,
                       in %data: !firrtl.uint<8>, out %clk_out: !firrtl.clock) {
    firrtl.matchingconnect %clk_out, %clk_in : !firrtl.clock
    %gc_clk, %gc_data = firrtl.instance gc @GrandChild(
      in clk: !firrtl.clock, in data: !firrtl.uint<8>)
    firrtl.matchingconnect %gc_clk, %gated_in : !firrtl.clock
    firrtl.matchingconnect %gc_data, %data : !firrtl.uint<8>
  }

  // CHECK-LABEL: firrtl.module @UngatedOutputWithGatedInput
  firrtl.module @UngatedOutputWithGatedInput(in %clk: !firrtl.clock,
                                             in %en: !firrtl.uint<1>,
                                             in %en2: !firrtl.uint<1>,
                                             in %data: !firrtl.uint<8>) {
    %g = firrtl.int.clock_gate %clk, %en

    // CHECK: firrtl.instance c1 @Child
    %c1_gated_in, %c1_clk_in, %c1_data, %c1_clk_out = firrtl.instance c1 @Child(
      in gated_in: !firrtl.clock, in clk_in: !firrtl.clock,
      in data: !firrtl.uint<8>, out clk_out: !firrtl.clock)
    firrtl.matchingconnect %c1_gated_in, %g : !firrtl.clock
    firrtl.matchingconnect %c1_clk_in, %clk : !firrtl.clock
    firrtl.matchingconnect %c1_data, %data : !firrtl.uint<8>

    %c2_gated_in, %c2_clk_in, %c2_data, %c2_clk_out = firrtl.instance c2 @Child(
      in gated_in: !firrtl.clock, in clk_in: !firrtl.clock,
      in data: !firrtl.uint<8>, out clk_out: !firrtl.clock)
    firrtl.matchingconnect %c2_gated_in, %g : !firrtl.clock
    firrtl.matchingconnect %c2_clk_in, %clk : !firrtl.clock
    firrtl.matchingconnect %c2_data, %data : !firrtl.uint<8>

    %g2 = firrtl.int.clock_gate %c1_clk_out, %en2
    // CHECK: %[[REG:.+]] = firrtl.reg %c1_clk_out
    %r = firrtl.reg %g2 : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %[[MUX:.+]] = firrtl.mux(%en2, %data, %[[REG]])
    // CHECK: firrtl.matchingconnect %[[REG]], %[[MUX]]
    firrtl.matchingconnect %r, %data : !firrtl.uint<8>
  }
}
