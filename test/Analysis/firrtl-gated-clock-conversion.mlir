// RUN: circt-opt %s -test-firrtl-gated-clock-conversion -split-input-file | FileCheck %s

// Each case below exercises one path of the GatedClockConversion utility:
// the gate's enable is sunk into the clocked op (register next-state mux or
// ref-force/release predicate) and the op's clock is rebound to the ungated
// base clock, eliminating the clock-gate from the clock path.


//===----------------------------------------------------------------------===//
// Cascaded clock gates: the enables are AND-reduced and the register is rebound
// all the way to the true ungated base clock.
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Multiple independent clock gates in one module: each register gets its own
// enable mux and is rebound to the shared base clock.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @TwoGates
firrtl.circuit "TwoGates" {
  firrtl.module @TwoGates(in %clk: !firrtl.clock, in %en1: !firrtl.uint<1>,
                          in %en2: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    // CHECK: %[[R1:.+]] = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %[[MUX1:.+]] = firrtl.mux(%en1, %d, %[[R1]])
    // CHECK: firrtl.matchingconnect %[[R1]], %[[MUX1]]
    // CHECK: %[[R2:.+]] = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %[[MUX2:.+]] = firrtl.mux(%en2, %d, %[[R2]])
    // CHECK: firrtl.matchingconnect %[[R2]], %[[MUX2]]
    %g1 = firrtl.int.clock_gate %clk, %en1
    %r1 = firrtl.reg %g1 : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r1, %d : !firrtl.uint<8>
    %g2 = firrtl.int.clock_gate %clk, %en2
    %r2 = firrtl.reg %g2 : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r2, %d : !firrtl.uint<8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Gated clock flowing through a wire alias before reaching the register: the
// utility looks through the wire/node/cast alias to find the gate.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @WireAlias
firrtl.circuit "WireAlias" {
  firrtl.module @WireAlias(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                           in %d: !firrtl.uint<8>) {
    // The base clock and enable are propagated onto wires; the register is
    // rebound to the base-clock wire and muxed with the enable wire.
    // CHECK: %[[CW:.+]] = firrtl.wire : !firrtl.clock
    // CHECK: %[[EW:.+]] = firrtl.wire : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %[[CW]], %clk
    // CHECK: firrtl.matchingconnect %[[EW]], %en
    // CHECK: %[[R:.+]] = firrtl.reg %[[CW]] : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %[[MUX:.+]] = firrtl.mux(%[[EW]], %d, %[[R]])
    // CHECK: firrtl.matchingconnect %[[R]], %[[MUX]]
    %g = firrtl.int.clock_gate %clk, %en
    %w = firrtl.wire : !firrtl.clock
    firrtl.matchingconnect %w, %g : !firrtl.clock
    %r = firrtl.reg %w : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// RefForceOp / RefReleaseOp on a gated clock: the clock is rebound to the base
// and the gate enable is AND-ed into the predicate.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @ForceRelease
firrtl.circuit "ForceRelease" {
  firrtl.module @ForceRelease(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                              in %cond: !firrtl.uint<1>, in %val: !firrtl.uint<8>,
                              in %d: !firrtl.uint<8>) {
    // The register on the gated clock gets the holding mux ...
    // CHECK: %[[R:.+]], %[[RREF:.+]] = firrtl.reg %clk forceable
    // CHECK: firrtl.mux(%en, %d, %[[R]])
    // ... and the force/release predicate is ANDed with the gate enable while
    // the clock is rebound to the base.
    // CHECK: %[[FAND:.+]] = firrtl.and %cond, %en
    // CHECK: firrtl.ref.force %clk, %[[FAND]], %[[RREF]], %val
    // CHECK: %[[RAND:.+]] = firrtl.and %cond, %en
    // CHECK: firrtl.ref.release %clk, %[[RAND]], %[[RREF]]
    %g = firrtl.int.clock_gate %clk, %en
    %r, %r_ref = firrtl.reg %g forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.ref.force %g, %cond, %r_ref, %val : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    firrtl.ref.release %g, %cond, %r_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Register clocked by a plain (ungated) clock: nothing to sink, so the IR is
// left unchanged (no mux, no rebind).
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Force on the ungated base clock while a register lives on the gated clock:
// only the register is muxed; the force predicate (on the free-running clock)
// is left untouched.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @ForceOnBase
firrtl.circuit "ForceOnBase" {
  firrtl.module @ForceOnBase(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                             in %cond: !firrtl.uint<1>, in %val: !firrtl.uint<8>,
                             in %d: !firrtl.uint<8>) {
    // CHECK: %[[R:.+]], %[[RREF:.+]] = firrtl.reg %clk forceable
    // CHECK: firrtl.mux(%en, %d, %[[R]])
    // The force is on %clk (ungated), so its predicate stays %cond.
    // CHECK: firrtl.ref.force %clk, %cond, %[[RREF]], %val
    // CHECK-NOT: firrtl.and
    %g = firrtl.int.clock_gate %clk, %en
    %r, %r_ref = firrtl.reg %g forceable : !firrtl.clock, !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    firrtl.ref.force %clk, %cond, %r_ref, %val : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Cross-module: the clock is gated in the parent and flows INTO a child module
// where the register lives.  A (base, enable) input-port pair is inserted on
// the child and driven from the parent's materialized values.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @Child
// CHECK-SAME:    in %[[CBASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk]]: !firrtl.clock
// CHECK-SAME:    in %[[CEN:[A-Za-z0-9_]*_gatedClock_enable_clk]]: !firrtl.uint<1>
// CHECK:         %[[R:.+]] = firrtl.reg %[[CBASE]] : !firrtl.clock, !firrtl.uint<8>
// CHECK:         %[[MUX:.+]] = firrtl.mux(%[[CEN]], %d, %[[R]])
// CHECK:         firrtl.matchingconnect %[[R]], %[[MUX]]

// CHECK-LABEL: firrtl.module @ClockInToChild
// CHECK:         %[[GATE:.+]] = firrtl.int.clock_gate %clk, %en
// CHECK:         firrtl.instance c @Child
// CHECK:         firrtl.matchingconnect %c__gatedClock_baseClock_clk
// CHECK:         firrtl.matchingconnect %c__gatedClock_enable_clk
firrtl.circuit "ClockInToChild" {
  firrtl.module @Child(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
  firrtl.module @ClockInToChild(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                                in %d: !firrtl.uint<8>) {
    %g = firrtl.int.clock_gate %clk, %en
    %c_clk, %c_d = firrtl.instance c @Child(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c_clk, %g : !firrtl.clock
    firrtl.matchingconnect %c_d, %d : !firrtl.uint<8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Cross-module: the clock is gated INSIDE a child module and flows OUT to a
// register in the parent.  (base, enable) output ports are inserted on the
// child so the parent register can rebind to the base and mux on the enable.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @Gen
// CHECK-SAME:    out %[[GBASE:[A-Za-z0-9_]*_gatedClock_baseClock_gclk]]: !firrtl.clock
// CHECK-SAME:    out %[[GEN:[A-Za-z0-9_]*_gatedClock_enable_gclk]]: !firrtl.uint<1>

// CHECK-LABEL: firrtl.module @ClockOutOfChild
// CHECK:         firrtl.instance c @Gen
// The parent register rebinds to the child's base-clock output port and muxes
// on the child's enable output port.
// CHECK:         %[[R:.+]] = firrtl.reg %c__gatedClock_baseClock_gclk : !firrtl.clock, !firrtl.uint<8>
// CHECK:         %[[MUX:.+]] = firrtl.mux(%c__gatedClock_enable_gclk, %d, %[[R]])
// CHECK:         firrtl.matchingconnect %[[R]], %[[MUX]]
firrtl.circuit "ClockOutOfChild" {
  firrtl.module @Gen(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                     out %gclk: !firrtl.clock) {
    %g = firrtl.int.clock_gate %clk, %en
    firrtl.matchingconnect %gclk, %g : !firrtl.clock
  }
  firrtl.module @ClockOutOfChild(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                                 in %d: !firrtl.uint<8>) {
    %c_clk, %c_en, %c_gclk = firrtl.instance c @Gen(in clk: !firrtl.clock, in en: !firrtl.uint<1>, out gclk: !firrtl.clock)
    firrtl.matchingconnect %c_clk, %clk : !firrtl.clock
    firrtl.matchingconnect %c_en, %en : !firrtl.uint<1>
    %r = firrtl.reg %c_gclk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Multi-level hierarchy: the clock is gated at the top and flows down through
// two levels of instances; the leaf register is rebound to the threaded base
// clock and muxed on the threaded enable.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @Leaf
// CHECK-SAME:    in %[[LBASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk]]: !firrtl.clock
// CHECK-SAME:    in %[[LEN:[A-Za-z0-9_]*_gatedClock_enable_clk]]: !firrtl.uint<1>
// CHECK:         %[[R:.+]] = firrtl.reg %[[LBASE]] : !firrtl.clock, !firrtl.uint<8>
// CHECK:         firrtl.mux(%[[LEN]], %data, %[[R]])

// CHECK-LABEL: firrtl.module @Mid
// CHECK:         firrtl.instance leaf @Leaf

// CHECK-LABEL: firrtl.module @Top
// CHECK:         %[[GATE:.+]] = firrtl.int.clock_gate %clk, %enable
// CHECK:         firrtl.instance mid @Mid
firrtl.circuit "Top" {
  firrtl.module @Leaf(in %clk: !firrtl.clock, in %enable: !firrtl.uint<1>,
                      in %data: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %data : !firrtl.uint<8>
  }
  firrtl.module @Mid(in %clk: !firrtl.clock, in %enable: !firrtl.uint<1>,
                     in %data: !firrtl.uint<8>) {
    %leaf_clk, %leaf_enable, %leaf_data = firrtl.instance leaf @Leaf(in clk: !firrtl.clock, in enable: !firrtl.uint<1>, in data: !firrtl.uint<8>)
    firrtl.matchingconnect %leaf_clk, %clk : !firrtl.clock
    firrtl.matchingconnect %leaf_enable, %enable : !firrtl.uint<1>
    firrtl.matchingconnect %leaf_data, %data : !firrtl.uint<8>
  }
  firrtl.module @Top(in %clk: !firrtl.clock, in %enable: !firrtl.uint<1>,
                     in %data: !firrtl.uint<8>) {
    %g = firrtl.int.clock_gate %clk, %enable
    %mid_clk, %mid_enable, %mid_data = firrtl.instance mid @Mid(in clk: !firrtl.clock, in enable: !firrtl.uint<1>, in data: !firrtl.uint<8>)
    firrtl.matchingconnect %mid_clk, %g : !firrtl.clock
    firrtl.matchingconnect %mid_enable, %enable : !firrtl.uint<1>
    firrtl.matchingconnect %mid_data, %data : !firrtl.uint<8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Dual independent gated clocks flowing through a multi-level hierarchy:
// Two separate clock gates (fast and slow) at the top level flow down through
// two levels of instances to registers in the leaf module. Each clock gets its
// own (base, enable) port pair threaded through the hierarchy.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @DualClockLeaf
// CHECK-SAME:    in %clk_fast: !firrtl.clock
// CHECK-SAME:    in %clk_slow: !firrtl.clock
// CHECK-SAME:    in %data_fast: !firrtl.uint<16>
// CHECK-SAME:    in %data_slow: !firrtl.uint<16>
// CHECK-SAME:    in %[[SBASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk_slow]]: !firrtl.clock
// CHECK-SAME:    in %[[SEN:[A-Za-z0-9_]*_gatedClock_enable_clk_slow]]: !firrtl.uint<1>
// CHECK-SAME:    in %[[FBASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk_fast]]: !firrtl.clock
// CHECK-SAME:    in %[[FEN:[A-Za-z0-9_]*_gatedClock_enable_clk_fast]]: !firrtl.uint<1>
// CHECK:         %[[RFAST:.+]] = firrtl.reg %[[FBASE]] : !firrtl.clock, !firrtl.uint<16>
// CHECK:         %[[MUXFAST:.+]] = firrtl.mux(%[[FEN]], %data_fast, %[[RFAST]])
// CHECK:         firrtl.matchingconnect %[[RFAST]], %[[MUXFAST]]
// CHECK:         %[[RSLOW:.+]] = firrtl.reg %[[SBASE]] : !firrtl.clock, !firrtl.uint<16>
// CHECK:         %[[MUXSLOW:.+]] = firrtl.mux(%[[SEN]], %data_slow, %[[RSLOW]])
// CHECK:         firrtl.matchingconnect %[[RSLOW]], %[[MUXSLOW]]

// CHECK-LABEL: firrtl.module @DualClockMid
// CHECK:         firrtl.instance leaf @DualClockLeaf

// CHECK-LABEL: firrtl.module @DualClockTop
// CHECK:         %[[GFAST:.+]] = firrtl.int.clock_gate %clk, %fast_en
// CHECK:         %[[GSLOW:.+]] = firrtl.int.clock_gate %clk, %slow_en
// CHECK:         firrtl.instance mid @DualClockMid
firrtl.circuit "DualClockTop" {
  firrtl.module @DualClockLeaf(in %clk_fast: !firrtl.clock,
                               in %clk_slow: !firrtl.clock,
                               in %data_fast: !firrtl.uint<16>,
                               in %data_slow: !firrtl.uint<16>) {
    %r_fast = firrtl.reg %clk_fast : !firrtl.clock, !firrtl.uint<16>
    firrtl.matchingconnect %r_fast, %data_fast : !firrtl.uint<16>


    %r_slow = firrtl.reg %clk_slow : !firrtl.clock, !firrtl.uint<16>
    firrtl.matchingconnect %r_slow, %data_slow : !firrtl.uint<16>
  }

  firrtl.module @DualClockMid(in %clk_fast: !firrtl.clock,
                              in %clk_slow: !firrtl.clock,
                              in %data_fast: !firrtl.uint<16>,
                              in %data_slow: !firrtl.uint<16>) {
    %leaf_fast, %leaf_slow, %leaf_df, %leaf_ds = firrtl.instance leaf @DualClockLeaf(in clk_fast: !firrtl.clock, in clk_slow: !firrtl.clock, in data_fast: !firrtl.uint<16>, in data_slow: !firrtl.uint<16>)
    firrtl.matchingconnect %leaf_fast, %clk_fast : !firrtl.clock
    firrtl.matchingconnect %leaf_slow, %clk_slow : !firrtl.clock
    firrtl.matchingconnect %leaf_df, %data_fast : !firrtl.uint<16>
    firrtl.matchingconnect %leaf_ds, %data_slow : !firrtl.uint<16>
  }

  firrtl.module @DualClockTop(in %clk: !firrtl.clock,
                              in %fast_en: !firrtl.uint<1>,
                              in %slow_en: !firrtl.uint<1>,
                              in %data_fast: !firrtl.uint<16>,
                              in %data_slow: !firrtl.uint<16>) {
    %g_fast = firrtl.int.clock_gate %clk, %fast_en
    %g_slow = firrtl.int.clock_gate %clk, %slow_en

    %mid_fast, %mid_slow, %mid_df, %mid_ds = firrtl.instance mid @DualClockMid(in clk_fast: !firrtl.clock, in clk_slow: !firrtl.clock, in data_fast: !firrtl.uint<16>, in data_slow: !firrtl.uint<16>)
    firrtl.matchingconnect %mid_fast, %g_fast : !firrtl.clock
    firrtl.matchingconnect %mid_slow, %g_slow : !firrtl.clock
    firrtl.matchingconnect %mid_df, %data_fast : !firrtl.uint<16>
    firrtl.matchingconnect %mid_ds, %data_slow : !firrtl.uint<16>
  }
}

// -----

// CHECK-LABEL: firrtl.module @MultipleSink
// CHECK-SAME:    in %clk: !firrtl.clock
// CHECK-SAME:    in %data_s: !firrtl.sint<16>
// CHECK-SAME:    in %data_u: !firrtl.uint<32>
// CHECK-SAME:    in %[[CBASE:[A-Za-z0-9_]*]]: !firrtl.clock
// CHECK-SAME:    in %[[CEN:[A-Za-z0-9_]*]]: !firrtl.uint<1>
// CHECK:         %[[RS:.+]] = firrtl.reg %[[CBASE]] : !firrtl.clock, !firrtl.sint<16>
// CHECK:         %[[MUXS:.+]] = firrtl.mux(%[[CEN]], %data_s, %[[RS]]) : (!firrtl.uint<1>, !firrtl.sint<16>, !firrtl.sint<16>) -> !firrtl.sint<16>
// CHECK:         firrtl.matchingconnect %[[RS]], %[[MUXS]] : !firrtl.sint<16>
// CHECK:         %[[RU:.+]] = firrtl.reg %[[CBASE]] : !firrtl.clock, !firrtl.uint<32>
// CHECK:         %[[MUXU:.+]] = firrtl.mux(%[[CEN]], %data_u, %[[RU]]) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
// CHECK:         firrtl.matchingconnect %[[RU]], %[[MUXU]] : !firrtl.uint<32>

// CHECK-LABEL: firrtl.module @MultipleSinkParent
firrtl.circuit "MultipleSinkParent" {
  firrtl.module @MultipleSink(in %clk: !firrtl.clock,
                                    in %data_s: !firrtl.sint<16>,
                                    in %data_u: !firrtl.uint<32>) {
    %rs = firrtl.reg %clk : !firrtl.clock, !firrtl.sint<16>
    firrtl.matchingconnect %rs, %data_s : !firrtl.sint<16>
    %ru = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<32>
    firrtl.matchingconnect %ru, %data_u : !firrtl.uint<32>
  }

  firrtl.module @MultipleSinkParent(in %clk: !firrtl.clock,
                                in %en: !firrtl.uint<1>,
                                in %ds: !firrtl.sint<16>,
                                in %du: !firrtl.uint<32>) {
    %g = firrtl.int.clock_gate %clk, %en
    %child_clk, %child_ds, %child_du = firrtl.instance child @MultipleSink(in clk: !firrtl.clock, in data_s: !firrtl.sint<16>, in data_u: !firrtl.uint<32>)
    firrtl.matchingconnect %child_clk, %g : !firrtl.clock
    firrtl.matchingconnect %child_ds, %ds : !firrtl.sint<16>
    firrtl.matchingconnect %child_du, %du : !firrtl.uint<32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Clock gate with test_enable in hierarchical context
// - Parent creates a gated clock with test_enable
// - Clock flows into child module
// - Enable should be materialized as (enable | test_enable)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @ChildWithTestEnable
// CHECK-SAME:    in %clk: !firrtl.clock
// CHECK-SAME:    in %data: !firrtl.uint<8>
// CHECK-SAME:    in %[[CBASE:[A-Za-z0-9_]*]]: !firrtl.clock
// CHECK-SAME:    in %[[CEN:[A-Za-z0-9_]*]]: !firrtl.uint<1>
// CHECK:         %[[R:.+]] = firrtl.reg %[[CBASE]] : !firrtl.clock, !firrtl.uint<8>
// CHECK:         %[[MUX:.+]] = firrtl.mux(%[[CEN]], %data, %[[R]]) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
// CHECK:         firrtl.matchingconnect %[[R]], %[[MUX]] : !firrtl.uint<8>

// CHECK-LABEL: firrtl.module @TestEnableHierarchy
firrtl.circuit "TestEnableHierarchy" {
  firrtl.module @ChildWithTestEnable(in %clk: !firrtl.clock, in %data: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %data : !firrtl.uint<8>
  }

  firrtl.module @TestEnableHierarchy(in %clk: !firrtl.clock,
                                     in %en: !firrtl.uint<1>,
                                     in %test_en: !firrtl.uint<1>,
                                     in %d: !firrtl.uint<8>) {
    // The effective enable should be (en | test_en)
    // CHECK: %[[OR:.+]] = firrtl.or %en, %test_en
    %g = firrtl.int.clock_gate %clk, %en, %test_en
    %child_clk, %child_data = firrtl.instance child @ChildWithTestEnable(in clk: !firrtl.clock, in data: !firrtl.uint<8>)
    firrtl.matchingconnect %child_clk, %g : !firrtl.clock
    firrtl.matchingconnect %child_data, %d : !firrtl.uint<8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Deeply nested: layerblock inside when inside match
// - Tests if the transformation handles deeply nested instances correctly
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @DeepChild
// CHECK-SAME:    in %[[DBASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk]]: !firrtl.clock
// CHECK-SAME:    in %[[DEN:[A-Za-z0-9_]*_gatedClock_enable_clk]]: !firrtl.uint<1>
// CHECK:         %[[R:.+]] = firrtl.reg %[[DBASE]] : !firrtl.clock, !firrtl.uint<8>
// CHECK:         %[[MUX:.+]] = firrtl.mux(%[[DEN]], %data, %[[R]])
// CHECK:         firrtl.matchingconnect %[[R]], %[[MUX]]

// CHECK-LABEL: firrtl.module @DeeplyNested
firrtl.circuit "DeeplyNested" {
  firrtl.layer @A bind {}

  firrtl.module @DeepChild(in %clk: !firrtl.clock, in %data: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %data : !firrtl.uint<8>
  }

  firrtl.module @DeeplyNested(in %clk: !firrtl.clock,
                              in %en: !firrtl.uint<1>,
                              in %selector: !firrtl.enum<A: uint<8>, B: uint<8>>,
                              in %cond: !firrtl.uint<1>,
                              in %d: !firrtl.uint<8>) {
    // CHECK: %[[G:.+]] = firrtl.int.clock_gate %clk, %en
    %g = firrtl.int.clock_gate %clk, %en

    // CHECK: firrtl.match %selector
    firrtl.match %selector : !firrtl.enum<A: uint<8>, B: uint<8>> {
      case A(%arg0) {
        // CHECK: firrtl.when %cond
        firrtl.when %cond : !firrtl.uint<1> {
          // CHECK: firrtl.layerblock @A
          firrtl.layerblock @A {
            // CHECK: %{{.+}}, %{{.+}}, %[[INST_BASE:.+]], %[[INST_EN:.+]] = firrtl.instance deep_child @DeepChild
            // CHECK-SAME: in [[GBASE_NAME:[A-Za-z0-9_]*_gatedClock_baseClock_clk]]: !firrtl.clock
            // CHECK-SAME: in [[GEN_NAME:[A-Za-z0-9_]*_gatedClock_enable_clk]]: !firrtl.uint<1>
            %child_clk, %child_data = firrtl.instance deep_child @DeepChild(in clk: !firrtl.clock, in data: !firrtl.uint<8>)
            firrtl.matchingconnect %child_clk, %g : !firrtl.clock
            firrtl.matchingconnect %child_data, %arg0 : !firrtl.uint<8>
            // CHECK: firrtl.matchingconnect %[[INST_BASE]], %clk
            // CHECK: firrtl.matchingconnect %[[INST_EN]], %en
          }
        }
      }
      case B(%arg0) {
        // Empty case
      }
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Same module instantiated twice with different gated clocks
// - First instance uses gated clock on clk1
// - Second instance uses gated clock on clk2
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @FlexibleChild
// CHECK-SAME:    in %clk1: !firrtl.clock
// CHECK-SAME:    in %clk2: !firrtl.clock
// CHECK-SAME:    in %data: !firrtl.uint<8>
// CHECK-SAME:    in %[[CLK2_BASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk2]]: !firrtl.clock
// CHECK-SAME:    in %[[CLK2_EN:[A-Za-z0-9_]*_gatedClock_enable_clk2]]: !firrtl.uint<1>
// CHECK-SAME:    in %[[CLK1_BASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk1]]: !firrtl.clock
// CHECK-SAME:    in %[[CLK1_EN:[A-Za-z0-9_]*_gatedClock_enable_clk1]]: !firrtl.uint<1>
// CHECK:         %[[R1:.+]] = firrtl.reg %[[CLK1_BASE]] : !firrtl.clock, !firrtl.uint<8>
// CHECK:         %[[R2:.+]] = firrtl.reg %[[CLK2_BASE]] : !firrtl.clock, !firrtl.uint<8>
// CHECK:         %[[MUX1:.+]] = firrtl.mux(%[[CLK1_EN]], %data, %[[R1]])
// CHECK:         firrtl.matchingconnect %[[R1]], %[[MUX1]]
// CHECK:         %[[MUX2:.+]] = firrtl.mux(%[[CLK2_EN]], %data, %[[R2]])
// CHECK:         firrtl.matchingconnect %[[R2]], %[[MUX2]]

// CHECK-LABEL: firrtl.module @DifferentGating
// CHECK:         %[[C1:.+]] = firrtl.constant 1 : !firrtl.uint<1>
// CHECK:         %[[C1_0:.+]] = firrtl.constant 1 : !firrtl.uint<1>
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
    // CHECK: %[[G1:.+]] = firrtl.int.clock_gate %base_clk, %en1
    %g1 = firrtl.int.clock_gate %base_clk, %en1
    // CHECK: %[[G2:.+]] = firrtl.int.clock_gate %base_clk, %en2
    %g2 = firrtl.int.clock_gate %base_clk, %en2

    // First instance: gate on clk1, ungated on clk2
    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[I1_CLK2_BASE:.+]], %[[I1_CLK2_EN:.+]], %[[I1_CLK1_BASE:.+]], %[[I1_CLK1_EN:.+]] = firrtl.instance inst1 @FlexibleChild
    %inst1_clk1, %inst1_clk2, %inst1_data = firrtl.instance inst1 @FlexibleChild(
      in clk1: !firrtl.clock,
      in clk2: !firrtl.clock,
      in data: !firrtl.uint<8>)
    // CHECK: firrtl.matchingconnect %{{.+}}, %[[G1]]
    firrtl.matchingconnect %inst1_clk1, %g1 : !firrtl.clock
    // CHECK: firrtl.matchingconnect %{{.+}}, %base_clk
    firrtl.matchingconnect %inst1_clk2, %base_clk : !firrtl.clock
    // CHECK: firrtl.matchingconnect %{{.+}}, %d
    firrtl.matchingconnect %inst1_data, %d : !firrtl.uint<8>

    // Second instance: ungated on clk1, gate on clk2
    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[I2_CLK2_BASE:.+]], %[[I2_CLK2_EN:.+]], %[[I2_CLK1_BASE:.+]], %[[I2_CLK1_EN:.+]] = firrtl.instance inst2 @FlexibleChild
    %inst2_clk1, %inst2_clk2, %inst2_data = firrtl.instance inst2 @FlexibleChild(
      in clk1: !firrtl.clock,
      in clk2: !firrtl.clock,
      in data: !firrtl.uint<8>)
    // CHECK: firrtl.matchingconnect %{{.+}}, %base_clk
    firrtl.matchingconnect %inst2_clk1, %base_clk : !firrtl.clock
    // CHECK: firrtl.matchingconnect %{{.+}}, %[[G2]]
    firrtl.matchingconnect %inst2_clk2, %g2 : !firrtl.clock
    // CHECK: firrtl.matchingconnect %{{.+}}, %d
    firrtl.matchingconnect %inst2_data, %d : !firrtl.uint<8>
    // The new (base, enable) ports are connected after the original instance connects
    // Instance1: ungated clk2 connects
    // CHECK: firrtl.matchingconnect %[[I1_CLK2_BASE]], %base_clk
    // CHECK: firrtl.matchingconnect %[[I1_CLK2_EN]], %[[C1_0]]
    // Instance2: ungated clk1 connects
    // CHECK: firrtl.matchingconnect %[[I2_CLK1_BASE]], %base_clk
    // CHECK: firrtl.matchingconnect %[[I2_CLK1_EN]], %[[C1]]
    // Instance2: gated clk2 connects
    // CHECK: firrtl.matchingconnect %[[I2_CLK2_BASE]], %base_clk
    // CHECK: firrtl.matchingconnect %[[I2_CLK2_EN]], %en2
    // Instance1: gated clk1 connects
    // CHECK: firrtl.matchingconnect %[[I1_CLK1_BASE]], %base_clk
    // CHECK: firrtl.matchingconnect %[[I1_CLK1_EN]], %en1
  }
}


// -----

//===----------------------------------------------------------------------===//
// Instance output port connected to wire outside when block
// - Tests if transformation handles instance outputs from nested blocks
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @Producer
// CHECK-SAME:    in %clk: !firrtl.clock
// CHECK-SAME:    out %out_clk: !firrtl.clock
// CHECK-SAME:    in %[[PROD_BASE_IN:[A-Za-z0-9_]*_gatedClock_baseClock_clk]]: !firrtl.clock
// CHECK-SAME:    in %[[PROD_EN_IN:[A-Za-z0-9_]*_gatedClock_enable_clk]]: !firrtl.uint<1>
// CHECK-SAME:    out %[[PROD_BASE_OUT:[A-Za-z0-9_]*_gatedClock_baseClock_out_clk]]: !firrtl.clock
// CHECK-SAME:    out %[[PROD_EN_OUT:[A-Za-z0-9_]*_gatedClock_enable_out_clk]]: !firrtl.uint<1>
// CHECK:         firrtl.matchingconnect %out_clk, %clk
// CHECK:         firrtl.matchingconnect %[[PROD_BASE_OUT]], %[[PROD_BASE_IN]]
// CHECK:         firrtl.matchingconnect %[[PROD_EN_OUT]], %[[PROD_EN_IN]]

// CHECK-LABEL: firrtl.module @Consumer
// CHECK-SAME:    in %clk: !firrtl.clock
// CHECK-SAME:    in %data: !firrtl.uint<8>
// CHECK-SAME:    in %[[CONS_BASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk]]: !firrtl.clock
// CHECK-SAME:    in %[[CONS_EN:[A-Za-z0-9_]*_gatedClock_enable_clk]]: !firrtl.uint<1>
// CHECK:         %[[R:.+]] = firrtl.reg %[[CONS_BASE]] : !firrtl.clock, !firrtl.uint<8>
// CHECK:         %[[MUX:.+]] = firrtl.mux(%[[CONS_EN]], %data, %[[R]])
// CHECK:         firrtl.matchingconnect %[[R]], %[[MUX]]

// CHECK-LABEL: firrtl.module @InstanceOutput
firrtl.circuit "InstanceOutput" {
  firrtl.module @Producer(in %clk: !firrtl.clock, out %out_clk: !firrtl.clock) {
    firrtl.matchingconnect %out_clk, %clk : !firrtl.clock
  }

  firrtl.module @Consumer(in %clk: !firrtl.clock, in %data: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %data : !firrtl.uint<8>
  }

  firrtl.module @InstanceOutput(in %base_clk: !firrtl.clock,
                                in %en: !firrtl.uint<1>,
                                in %cond: !firrtl.uint<1>,
                                in %d: !firrtl.uint<8>) {
    // CHECK: %[[WIRE_BASE:.+]] = firrtl.wire : !firrtl.clock
    // CHECK: %[[WIRE_EN:.+]] = firrtl.wire : !firrtl.uint<1>
    // CHECK: %[[GATE:.+]] = firrtl.int.clock_gate %base_clk, %en
    %gated = firrtl.int.clock_gate %base_clk, %en
    // CHECK: %[[OUT_WIRE:.+]] = firrtl.wire : !firrtl.clock
    %out_wire = firrtl.wire : !firrtl.clock

    // CHECK: firrtl.when %cond
    firrtl.when %cond : !firrtl.uint<1> {
      // CHECK: %{{.+}}, %{{.+}}, %[[P_BASE_IN:.+]], %[[P_EN_IN:.+]], %[[P_BASE_OUT:.+]], %[[P_EN_OUT:.+]] = firrtl.instance producer @Producer
      %p_clk, %p_out = firrtl.instance producer @Producer(in clk: !firrtl.clock, out out_clk: !firrtl.clock)
      // CHECK: firrtl.matchingconnect %[[WIRE_EN]], %[[P_EN_OUT]]
      // CHECK: firrtl.matchingconnect %[[WIRE_BASE]], %[[P_BASE_OUT]]
      // CHECK: firrtl.matchingconnect %{{.+}}, %[[GATE]]
      firrtl.matchingconnect %p_clk, %gated : !firrtl.clock
      firrtl.matchingconnect %out_wire, %p_out : !firrtl.clock
      // CHECK: firrtl.matchingconnect %[[P_BASE_IN]], %base_clk
      // CHECK: firrtl.matchingconnect %[[P_EN_IN]], %en
    }

    // CHECK: %{{.+}}, %{{.+}}, %[[C_BASE:.+]], %[[C_EN:.+]] = firrtl.instance consumer @Consumer
    %c_clk, %c_data = firrtl.instance consumer @Consumer(in clk: !firrtl.clock, in data: !firrtl.uint<8>)
    // CHECK: firrtl.matchingconnect %{{.+}}, %[[OUT_WIRE]]
    firrtl.matchingconnect %c_clk, %out_wire : !firrtl.clock
    // CHECK: firrtl.matchingconnect %{{.+}}, %d
    firrtl.matchingconnect %c_data, %d : !firrtl.uint<8>
    // CHECK: firrtl.matchingconnect %[[C_BASE]], %[[WIRE_BASE]]
    // CHECK: firrtl.matchingconnect %[[C_EN]], %[[WIRE_EN]]
  }
}
