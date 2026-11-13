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
// CHECK-SAME:    in %[[FBASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk_fast]]: !firrtl.clock
// CHECK-SAME:    in %[[FEN:[A-Za-z0-9_]*_gatedClock_enable_clk_fast]]: !firrtl.uint<1>
// CHECK-SAME:    in %[[SBASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk_slow]]: !firrtl.clock
// CHECK-SAME:    in %[[SEN:[A-Za-z0-9_]*_gatedClock_enable_clk_slow]]: !firrtl.uint<1>
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

// An instance in a layerblock, whose clock is gated by an enable defined after
// the layerblock: the enable reaches the instance through a wire.
// CHECK-LABEL: firrtl.module private @LayerChild
// CHECK-LABEL: firrtl.module @EnableAfterLayerBlock
firrtl.circuit "EnableAfterLayerBlock" {
  firrtl.layer @A bind {}
  firrtl.module private @LayerChild(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
  firrtl.module @EnableAfterLayerBlock(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                                       in %d: !firrtl.uint<8>) {
    // CHECK: %[[EN_WIRE:.+]] = firrtl.wire : !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.clock
    // CHECK: firrtl.layerblock @A {
    // CHECK:   %{{.+}}, %{{.+}}, %[[BASE:.+]], %[[EN:.+]] = firrtl.instance a @LayerChild
    // CHECK:   firrtl.matchingconnect %[[BASE]], %clk : !firrtl.clock
    // CHECK:   firrtl.matchingconnect %[[EN]], %[[EN_WIRE]] : !firrtl.uint<1>
    // CHECK: }
    firrtl.layerblock @A {
      %a_clk, %a_d = firrtl.instance a @LayerChild(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
      firrtl.matchingconnect %a_clk, %w : !firrtl.clock
      firrtl.matchingconnect %a_d, %d : !firrtl.uint<8>
    }
    // CHECK: %[[NOT:.+]] = firrtl.not %en
    // CHECK: firrtl.matchingconnect %[[EN_WIRE]], %[[NOT]] : !firrtl.uint<1>
    %e = firrtl.not %en : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %g = firrtl.int.clock_gate %clk, %e
    firrtl.matchingconnect %w, %g : !firrtl.clock
  }
}

// -----

// A gate in a layerblock, on a clock gated by an enable defined after the
// layerblock: the enables are ANDed at the inner gate, with the outer enable
// reaching it through a wire.
// CHECK-LABEL: firrtl.module @GateInLayerBlock
firrtl.circuit "GateInLayerBlock" {
  firrtl.layer @A bind {}
  firrtl.module @GateInLayerBlock(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                                  in %en2: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    // CHECK: %[[EN_WIRE:.+]] = firrtl.wire : !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.clock
    // CHECK: firrtl.layerblock @A {
    // CHECK:   %[[NOT2:.+]] = firrtl.not %en2
    // CHECK:   %[[AND:.+]] = firrtl.and %[[EN_WIRE]], %[[NOT2]]
    // CHECK:   %[[R:.+]] = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    // CHECK:   %[[MUX:.+]] = firrtl.mux(%[[AND]], %d, %[[R]])
    // CHECK:   firrtl.matchingconnect %[[R]], %[[MUX]]
    // CHECK: }
    firrtl.layerblock @A {
      %n = firrtl.not %en2 : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %g2 = firrtl.int.clock_gate %w, %n
      %r = firrtl.reg %g2 : !firrtl.clock, !firrtl.uint<8>
      firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    }
    // CHECK: %[[NOT:.+]] = firrtl.not %en
    // CHECK: firrtl.matchingconnect %[[EN_WIRE]], %[[NOT]] : !firrtl.uint<1>
    %e = firrtl.not %en : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %g = firrtl.int.clock_gate %clk, %e
    firrtl.matchingconnect %w, %g : !firrtl.clock
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
// CHECK-LABEL: firrtl.module private @MultiIOClock
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
  firrtl.module private @MultiIOClock(in %clk_in1: !firrtl.clock, in %clk_in2: !firrtl.clock,
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
// CHECK-LABEL: firrtl.module private @ClockGen
// CHECK-SAME:    in %[[BASE_IN:[A-Za-z0-9_]*_gatedClock_baseClock_clk_in]]: !firrtl.clock
// CHECK-SAME:    in %[[EN_IN:[A-Za-z0-9_]*_gatedClock_enable_clk_in]]: !firrtl.uint<1>
// CHECK-SAME:    out %[[BASE_OUT:[A-Za-z0-9_]*_gatedClock_baseClock_clk_out]]: !firrtl.clock
// CHECK-SAME:    out %[[EN_OUT:[A-Za-z0-9_]*_gatedClock_enable_clk_out]]: !firrtl.uint<1>
firrtl.circuit "Top" {
  firrtl.module private @ClockGen(in %clk_in: !firrtl.clock, in %en: !firrtl.uint<1>, out %clk_out: !firrtl.clock) {
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
// CHECK-LABEL: firrtl.module private @ClockGen
// CHECK-SAME:    in %[[CG_BI:[A-Za-z0-9_]*_gatedClock_baseClock_clk_in]]: !firrtl.clock
// CHECK-SAME:    in %[[CG_EI:[A-Za-z0-9_]*_gatedClock_enable_clk_in]]: !firrtl.uint<1>
// CHECK-SAME:    out %[[CG_BO:[A-Za-z0-9_]*_gatedClock_baseClock_clk_out]]: !firrtl.clock
// CHECK-SAME:    out %[[CG_EO:[A-Za-z0-9_]*_gatedClock_enable_clk_out]]: !firrtl.uint<1>
firrtl.circuit "Top" {
  firrtl.module private @ClockGen(in %clk_in: !firrtl.clock, in %en: !firrtl.uint<1>, out %clk_out: !firrtl.clock) {
    // CHECK: %[[AND:.+]] = firrtl.and %[[CG_EI]], %en
    // CHECK: firrtl.matchingconnect %[[CG_BO]], %[[CG_BI]]
    // CHECK: firrtl.matchingconnect %[[CG_EO]], %[[AND]]
    %g = firrtl.int.clock_gate %clk_in, %en
    firrtl.matchingconnect %clk_out, %g : !firrtl.clock
  }

  // CHECK-LABEL: firrtl.module private @Mid
  // CHECK-SAME:    in %[[M_BI:[A-Za-z0-9_]*_gatedClock_baseClock_clk_in]]: !firrtl.clock
  // CHECK-SAME:    in %[[M_EI:[A-Za-z0-9_]*_gatedClock_enable_clk_in]]: !firrtl.uint<1>
  // CHECK-SAME:    out %[[M_BO:[A-Za-z0-9_]*_gatedClock_baseClock_clk_out]]: !firrtl.clock
  // CHECK-SAME:    out %[[M_EO:[A-Za-z0-9_]*_gatedClock_enable_clk_out]]: !firrtl.uint<1>
  firrtl.module private @Mid(in %clk_in: !firrtl.clock, in %en: !firrtl.uint<1>, out %clk_out: !firrtl.clock) {
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

// `c1` is re-created for `Child`'s new ports, so the register must be clocked
// by the new instance's ungated output.
firrtl.circuit "UngatedOutputWithGatedInput" {
  // CHECK-LABEL: firrtl.module private @GrandChild
  // CHECK-SAME:    in %[[GC_BASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk]]: !firrtl.clock
  // CHECK-SAME:    in %[[GC_EN:[A-Za-z0-9_]*_gatedClock_enable_clk]]: !firrtl.uint<1>
  // CHECK:         %[[GC_REG:.+]] = firrtl.reg %[[GC_BASE]]
  // CHECK:         %[[GC_MUX:.+]] = firrtl.mux(%[[GC_EN]], %data, %[[GC_REG]])
  // CHECK:         firrtl.matchingconnect %[[GC_REG]], %[[GC_MUX]]
  firrtl.module private @GrandChild(in %clk: !firrtl.clock, in %data: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %data : !firrtl.uint<8>
  }

  // CHECK-LABEL: firrtl.module private @Child
  // CHECK-SAME:    in %[[CHILD_BASE:[A-Za-z0-9_]*_gatedClock_baseClock_gated_in]]: !firrtl.clock
  // CHECK-SAME:    in %[[CHILD_EN:[A-Za-z0-9_]*_gatedClock_enable_gated_in]]: !firrtl.uint<1>
  // CHECK:         firrtl.matchingconnect %clk_out, %clk_in
  // CHECK:         firrtl.matchingconnect %[[GC_BASE_IN:.+]], %[[CHILD_BASE]]
  // CHECK:         firrtl.matchingconnect %[[GC_EN_IN:.+]], %[[CHILD_EN]]
  firrtl.module private @Child(in %gated_in: !firrtl.clock, in %clk_in: !firrtl.clock,
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
  firrtl.layer @TestLayer bind {}

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
    // CHECK: %[[GATE:.+]] = firrtl.int.clock_gate %base_clk, %en
    %gated = firrtl.int.clock_gate %base_clk, %en

    // CHECK: firrtl.layerblock @TestLayer
    firrtl.layerblock @TestLayer {
      // CHECK: %[[OUT_WIRE:.+]] = firrtl.wire : !firrtl.clock
      %out_wire = firrtl.wire : !firrtl.clock
      // CHECK: %[[WIRE_BASE:.+]] = firrtl.wire : !firrtl.clock
      %wire_base = firrtl.wire : !firrtl.clock
      // CHECK: %[[WIRE_EN:.+]] = firrtl.wire : !firrtl.uint<1>
      %wire_en = firrtl.wire : !firrtl.uint<1>

      // CHECK: %{{.+}}, %{{.+}}, %[[P_BASE_IN:.+]], %[[P_EN_IN:.+]], %[[P_BASE_OUT:.+]], %[[P_EN_OUT:.+]] = firrtl.instance producer @Producer
      %p_clk, %p_out = firrtl.instance producer @Producer(in clk: !firrtl.clock, out out_clk: !firrtl.clock)
      firrtl.matchingconnect %p_clk, %gated : !firrtl.clock
      firrtl.matchingconnect %out_wire, %p_out : !firrtl.clock
      firrtl.matchingconnect %wire_base, %p_out : !firrtl.clock
      %c1 = firrtl.constant 1 : !firrtl.uint<1>
      firrtl.matchingconnect %wire_en, %c1 : !firrtl.uint<1>

      // CHECK: %{{.+}}, %{{.+}}, %[[C_BASE:.+]], %[[C_EN:.+]] = firrtl.instance consumer @Consumer
      %c_clk, %c_data = firrtl.instance consumer @Consumer(in clk: !firrtl.clock, in data: !firrtl.uint<8>)
      firrtl.matchingconnect %c_clk, %out_wire : !firrtl.clock
      firrtl.matchingconnect %c_data, %d : !firrtl.uint<8>
      firrtl.matchingconnect %c_clk, %wire_base : !firrtl.clock
      firrtl.matchingconnect %c_data, %d : !firrtl.uint<8>
      // CHECK: firrtl.matchingconnect %[[P_BASE_IN]], %base_clk
      // CHECK: firrtl.matchingconnect %[[P_EN_IN]], %en
      // CHECK: firrtl.matchingconnect %[[C_BASE]], %[[P_BASE_OUT]]
      // CHECK: firrtl.matchingconnect %[[C_EN]], %[[P_EN_OUT]]
    }
  }
}

// -----

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
// Same module instantiated twice: gated clock into one instance, ungated into another
// This tests that the pass correctly inserts (base, enable) ports for the gated
// instance and uses constant 1 for the enable on the ungated instance.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @FlexibleClockModule
// CHECK-SAME:    in %clk: !firrtl.clock
// CHECK-SAME:    in %data: !firrtl.uint<8>
// CHECK-SAME:    in %[[BASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk]]: !firrtl.clock
// CHECK-SAME:    in %[[EN:[A-Za-z0-9_]*_gatedClock_enable_clk]]: !firrtl.uint<1>
firrtl.circuit "MixedGatingInstances" {
  firrtl.module @FlexibleClockModule(in %clk: !firrtl.clock, in %data: !firrtl.uint<8>) {
    // CHECK: %[[R:.+]] = firrtl.reg %[[BASE]] : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %[[MUX:.+]] = firrtl.mux(%[[EN]], %data, %[[R]])
    // CHECK: firrtl.matchingconnect %[[R]], %[[MUX]]
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %data : !firrtl.uint<8>
  }

  // CHECK-LABEL: firrtl.module @MixedGatingInstances
  firrtl.module @MixedGatingInstances(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    // CHECK: %[[C1:.+]] = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: firrtl.int.clock_gate %clk, %en
    // CHECK: %{{.+}}, %{{.+}}, %[[I1_BASE:.+]], %[[I1_EN:.+]] = firrtl.instance inst1 @FlexibleClockModule
    // CHECK: %{{.+}}, %{{.+}}, %[[I2_BASE:.+]], %[[I2_EN:.+]] = firrtl.instance inst2 @FlexibleClockModule
    %g = firrtl.int.clock_gate %clk, %en

    // First instance: ungated clock
    %inst1_clk, %inst1_data = firrtl.instance inst1 @FlexibleClockModule(in clk: !firrtl.clock, in data: !firrtl.uint<8>)
    firrtl.matchingconnect %inst1_clk, %clk : !firrtl.clock
    firrtl.matchingconnect %inst1_data, %d : !firrtl.uint<8>

    // Second instance: gated clock
    %inst2_clk, %inst2_data = firrtl.instance inst2 @FlexibleClockModule(in clk: !firrtl.clock, in data: !firrtl.uint<8>)
    firrtl.matchingconnect %inst2_clk, %g : !firrtl.clock
    firrtl.matchingconnect %inst2_data, %d : !firrtl.uint<8>
    // CHECK: firrtl.matchingconnect %[[I1_BASE]], %clk
    // CHECK: firrtl.matchingconnect %[[I1_EN]], %[[C1]]
    // CHECK: firrtl.matchingconnect %[[I2_BASE]], %clk
    // CHECK: firrtl.matchingconnect %[[I2_EN]], %en
  }
}

// -----

//===----------------------------------------------------------------------===//
// Multiple different clocks to the same module: one gated, one ungated
// Tests that only the gated clock (clk2) gets (base, enable) port pair added.
// The ungated clock (clk1) remains unchanged and uses the original clock directly.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.module @DualClockConsumer
// CHECK-SAME:    in %clk1: !firrtl.clock
// CHECK-SAME:    in %clk2: !firrtl.clock
// CHECK-SAME:    in %data: !firrtl.uint<16>
// CHECK-SAME:    in %[[CLK2_BASE:[A-Za-z0-9_]*_gatedClock_baseClock_clk2]]: !firrtl.clock
// CHECK-SAME:    in %[[CLK2_EN:[A-Za-z0-9_]*_gatedClock_enable_clk2]]: !firrtl.uint<1>
firrtl.circuit "MixedClockDomains" {
  firrtl.module @DualClockConsumer(in %clk1: !firrtl.clock, in %clk2: !firrtl.clock, in %data: !firrtl.uint<16>) {
    // r1 remains on the original ungated clk1
    // CHECK: %[[R1:.+]] = firrtl.reg %clk1 : !firrtl.clock, !firrtl.uint<16>
    // CHECK-NEXT: firrtl.matchingconnect %[[R1]], %data
    %r1 = firrtl.reg %clk1 : !firrtl.clock, !firrtl.uint<16>
    firrtl.matchingconnect %r1, %data : !firrtl.uint<16>

    // r2 is rebound to the base clock and gets a mux with the enable
    // CHECK: %[[R2:.+]] = firrtl.reg %[[CLK2_BASE]] : !firrtl.clock, !firrtl.uint<16>
    // CHECK: %[[MUX2:.+]] = firrtl.mux(%[[CLK2_EN]], %data, %[[R2]])
    // CHECK: firrtl.matchingconnect %[[R2]], %[[MUX2]]
    %r2 = firrtl.reg %clk2 : !firrtl.clock, !firrtl.uint<16>
    firrtl.matchingconnect %r2, %data : !firrtl.uint<16>
  }

  // CHECK-LABEL: firrtl.module @MixedClockDomains
  firrtl.module @MixedClockDomains(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<16>) {
    // CHECK: firrtl.int.clock_gate %clk, %en
    %g = firrtl.int.clock_gate %clk, %en

    // Instance with one ungated (clk1) and one gated (clk2) clock
    // Only clk2 gets the gated clock ports
    // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %[[C2_BASE:.+]], %[[C2_EN:.+]] = firrtl.instance consumer @DualClockConsumer
    %c_clk1, %c_clk2, %c_data = firrtl.instance consumer @DualClockConsumer(in clk1: !firrtl.clock, in clk2: !firrtl.clock, in data: !firrtl.uint<16>)

    // clk1 is ungated
    firrtl.matchingconnect %c_clk1, %clk : !firrtl.clock
    // clk2 is gated
    firrtl.matchingconnect %c_clk2, %g : !firrtl.clock
    firrtl.matchingconnect %c_data, %d : !firrtl.uint<16>

    // Only clk2's base and enable are connected (clk1 needs no extra ports)
    // CHECK: firrtl.matchingconnect %[[C2_BASE]], %clk
    // CHECK: firrtl.matchingconnect %[[C2_EN]], %en
  }
}
