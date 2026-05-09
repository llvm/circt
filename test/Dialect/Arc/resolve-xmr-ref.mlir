// RUN: circt-opt --arc-resolve-xmr=lower-blackbox-internal-to-zero %s | FileCheck %s

module {
  hw.hierpath @intSigPath [@Top::@mid_inst, @Mid::@leaf_li, @LeafInternal::@signal]
  hw.hierpath @intPortPath [@Top::@mid_inst, @Mid::@leaf_li, @LeafInternal::@in_data]
  hw.hierpath @extPortPath [@Top::@mid_inst, @Mid::@cg_inst, @ClockGate::@out_clk]
  hw.hierpath @extSigPath [@Top::@mid_inst, @Mid::@cg_inst, @ClockGate::@secret_node]
  hw.hierpath @extNoUsedPath [@Top::@mid_inst, @Mid::@cg_inst, @ClockGate::@noused_node]

  // Reuse checks: same source path should reuse the same bored ports.
  hw.hierpath @reusePathA [@TopReuse::@mid_inst_0, @MidReuse::@leaf_shared, @LeafReuse::@sig]
  hw.hierpath @reusePathB [@TopReuse::@mid_inst_1, @MidReuse::@leaf_shared, @LeafReuse::@sig]

  // Collision checks: same target symbol name on different path suffixes should
  // not be merged into one bored port.
  hw.hierpath @collisionA [@TopCollision::@mid_inst, @MidCollision::@leaf_a, @LeafA::@data]
  hw.hierpath @collisionB [@TopCollision::@mid_inst, @MidCollision::@leaf_b, @LeafB::@data]

  // Collision checks (same type): same target symbol name and type on different
  // suffix paths should still produce distinct bored ports.
  hw.hierpath @sameTypeCollisionA [@TopSameTypeCollision::@mid_inst, @MidSameTypeCollision::@leaf_a, @LeafSameTypeA::@sig]
  hw.hierpath @sameTypeCollisionB [@TopSameTypeCollision::@mid_inst, @MidSameTypeCollision::@leaf_b, @LeafSameTypeB::@sig]

  // CHECK-LABEL: hw.module @LeafInternal
  // CHECK-SAME: out xmr_bored_signal_{{[0-9]+}} : i32
  hw.module @LeafInternal(in %in_data : i32 {hw.exportPort = #hw<innerSym@in_data>}) {
    %c42_i32 = hw.constant 42 : i32
    %wire_sig = hw.wire %c42_i32 sym @signal : i32
    // CHECK: hw.output %wire_sig : i32
    hw.output
  }

  // CHECK-LABEL: hw.module.extern @ClockGate
  hw.module.extern @ClockGate(out out_clk : i1 {hw.exportPort = #hw<innerSym@out_clk>})

  // CHECK-LABEL: hw.module @Mid
  // CHECK-SAME: out xmr_bored_signal_{{[0-9]+}} : i32
  // CHECK-SAME: out xmr_bored_in_data_{{[0-9]+}} : i32
  // CHECK-SAME: out xmr_bored_out_clk_{{[0-9]+}} : i1
  // CHECK-SAME: out xmr_bored_secret_node_{{[0-9]+}} : i1
  hw.module @Mid() {
    %c0 = hw.constant 0 : i32
    // CHECK: %[[LI_OUT:.+]] = hw.instance "leaf_li" sym @leaf_li @LeafInternal(in_data: %{{.+}}: i32) -> (xmr_bored_signal_{{[0-9]+}}: i32)
    hw.instance "leaf_li" sym @leaf_li @LeafInternal(in_data: %c0 : i32) -> ()

    // CHECK: %[[FALSE:.+]] = hw.constant false
    // CHECK: %[[CG_OUT:.+]] = hw.instance "cg_inst" sym @cg_inst @ClockGate() -> (out_clk: i1)
    %cg_clk = hw.instance "cg_inst" sym @cg_inst @ClockGate() -> (out_clk: i1)

    // CHECK: hw.output %[[LI_OUT]], %{{.+}}, %[[CG_OUT]], %[[FALSE]] : i32, i32, i1, i1
    hw.output
  }

  // CHECK-LABEL: hw.module @Top
  // CHECK: %[[MID_OUT_SIG:.+]], %[[MID_OUT_PORT:.+]], %[[MID_OUT_CLK:.+]], %[[MID_OUT_SEC:.+]] = hw.instance "mid_inst" sym @mid_inst @Mid() -> (xmr_bored_signal_{{[0-9]+}}: i32, xmr_bored_in_data_{{[0-9]+}}: i32, xmr_bored_out_clk_{{[0-9]+}}: i1, xmr_bored_secret_node_{{[0-9]+}}: i1)
  hw.module @Top(out out_int_sig : i32, out out_int_port : i32,
                 out out_ext_port : i1, out out_ext_sig : i1) {
    hw.instance "mid_inst" sym @mid_inst @Mid() -> ()

    %0 = sv.xmr.ref @intSigPath : !hw.inout<i32>
    %1 = sv.read_inout %0 : !hw.inout<i32>
    %2 = sv.xmr.ref @intPortPath : !hw.inout<i32>
    %3 = sv.read_inout %2 : !hw.inout<i32>
    %4 = sv.xmr.ref @extPortPath : !hw.inout<i1>
    %5 = sv.read_inout %4 : !hw.inout<i1>
    %6 = sv.xmr.ref @extSigPath : !hw.inout<i1>
    %7 = sv.read_inout %6 : !hw.inout<i1>

    // CHECK: hw.output %[[MID_OUT_SIG]], %[[MID_OUT_PORT]], %[[MID_OUT_CLK]], %[[MID_OUT_SEC]] : i32, i32, i1, i1
    hw.output %1, %3, %5, %7 : i32, i32, i1, i1
  }

  // CHECK-LABEL: hw.module @LeafReuse
  // CHECK-SAME: out xmr_bored_sig_{{[0-9]+}} : i8
  hw.module @LeafReuse() {
    %c = hw.constant 7 : i8
    %s = hw.wire %c sym @sig : i8
    // CHECK: hw.output %s : i8
    hw.output
  }

  // CHECK-LABEL: hw.module @MidReuse
  // CHECK-SAME: out xmr_bored_sig_{{[0-9]+}} : i8
  hw.module @MidReuse() {
    // CHECK: %[[L0:.+]] = hw.instance "leaf_shared" sym @leaf_shared @LeafReuse() -> (xmr_bored_sig_{{[0-9]+}}: i8)
    hw.instance "leaf_shared" sym @leaf_shared @LeafReuse() -> ()
    // CHECK: hw.output %[[L0]] : i8
    hw.output
  }

  // Same downstream path suffixes should reuse a single bored output in
  // @MidReuse even when reached via different top-level instances.
  // CHECK-LABEL: hw.module @TopReuse
  // CHECK: %[[M0:.+]] = hw.instance "mid_inst_0" sym @mid_inst_0 @MidReuse() -> (xmr_bored_sig_{{[0-9]+}}: i8)
  // CHECK: %[[M1:.+]] = hw.instance "mid_inst_1" sym @mid_inst_1 @MidReuse() -> (xmr_bored_sig_{{[0-9]+}}: i8)
  // CHECK: hw.output %[[M0]], %[[M1]], %[[M0]], %[[M1]] : i8, i8, i8, i8
  hw.module @TopReuse(out a0 : i8, out a1 : i8, out b0 : i8, out b1 : i8) {
    hw.instance "mid_inst_0" sym @mid_inst_0 @MidReuse() -> ()
    hw.instance "mid_inst_1" sym @mid_inst_1 @MidReuse() -> ()

    %ra0 = sv.xmr.ref @reusePathA : !hw.inout<i8>
    %rb0 = sv.xmr.ref @reusePathB : !hw.inout<i8>
    %ra1 = sv.xmr.ref @reusePathA : !hw.inout<i8>
    %rb1 = sv.xmr.ref @reusePathB : !hw.inout<i8>
    %va0 = sv.read_inout %ra0 : !hw.inout<i8>
    %vb0 = sv.read_inout %rb0 : !hw.inout<i8>
    %va1 = sv.read_inout %ra1 : !hw.inout<i8>
    %vb1 = sv.read_inout %rb1 : !hw.inout<i8>
    hw.output %va0, %vb0, %va1, %vb1 : i8, i8, i8, i8
  }

  // CHECK-LABEL: hw.module @LeafA
  // CHECK-SAME: out xmr_bored_data_{{[0-9]+}} : i1
  hw.module @LeafA() {
    %c = hw.constant true
    %s = hw.wire %c sym @data : i1
    hw.output
  }

  // CHECK-LABEL: hw.module @LeafB
  // CHECK-SAME: out xmr_bored_data_{{[0-9]+}} : i4
  hw.module @LeafB() {
    %c = hw.constant 3 : i4
    %s = hw.wire %c sym @data : i4
    hw.output
  }

  // Two different suffix paths share target symbol name `data` but differ in
  // type. They must not collide into one bored output.
  // CHECK-LABEL: hw.module @MidCollision
  // CHECK-SAME: out xmr_bored_data_{{[0-9]+}} : i1
  // CHECK-SAME: out xmr_bored_data_{{[0-9]+}} : i4
  // CHECK: %[[LA:.+]] = hw.instance "leaf_a" sym @leaf_a @LeafA() -> (xmr_bored_data_{{[0-9]+}}: i1)
  // CHECK: %[[LB:.+]] = hw.instance "leaf_b" sym @leaf_b @LeafB() -> (xmr_bored_data_{{[0-9]+}}: i4)
  // CHECK: hw.output %[[LA]], %[[LB]] : i1, i4
  hw.module @MidCollision() {
    hw.instance "leaf_a" sym @leaf_a @LeafA() -> ()
    hw.instance "leaf_b" sym @leaf_b @LeafB() -> ()
    hw.output
  }

  // CHECK-LABEL: hw.module @TopCollision
  // CHECK: %[[MC0:.+]], %[[MC1:.+]] = hw.instance "mid_inst" sym @mid_inst @MidCollision() -> (xmr_bored_data_{{[0-9]+}}: i1, xmr_bored_data_{{[0-9]+}}: i4)
  // CHECK: hw.output %[[MC0]], %[[MC1]] : i1, i4
  hw.module @TopCollision(out out_a : i1, out out_b : i4) {
    hw.instance "mid_inst" sym @mid_inst @MidCollision() -> ()
    %ca = sv.xmr.ref @collisionA : !hw.inout<i1>
    %cb = sv.xmr.ref @collisionB : !hw.inout<i4>
    %va = sv.read_inout %ca : !hw.inout<i1>
    %vb = sv.read_inout %cb : !hw.inout<i4>
    hw.output %va, %vb : i1, i4
  }

  // CHECK-LABEL: hw.module @LeafSameTypeA
  // CHECK-SAME: out xmr_bored_sig_{{[0-9]+}} : i1
  hw.module @LeafSameTypeA() {
    %c = hw.constant true
    %s = hw.wire %c sym @sig : i1
    hw.output
  }

  // CHECK-LABEL: hw.module @LeafSameTypeB
  // CHECK-SAME: out xmr_bored_sig_{{[0-9]+}} : i1
  hw.module @LeafSameTypeB() {
    %c = hw.constant false
    %s = hw.wire %c sym @sig : i1
    hw.output
  }

  // CHECK-LABEL: hw.module @MidSameTypeCollision
  // CHECK-SAME: out xmr_bored_sig_{{[0-9]+}} : i1
  // CHECK-SAME: out xmr_bored_sig_{{[0-9]+}} : i1
  // CHECK: %[[LSA:.+]] = hw.instance "leaf_a" sym @leaf_a @LeafSameTypeA() -> (xmr_bored_sig_{{[0-9]+}}: i1)
  // CHECK: %[[LSB:.+]] = hw.instance "leaf_b" sym @leaf_b @LeafSameTypeB() -> (xmr_bored_sig_{{[0-9]+}}: i1)
  // CHECK: hw.output %[[LSA]], %[[LSB]] : i1, i1
  hw.module @MidSameTypeCollision() {
    hw.instance "leaf_a" sym @leaf_a @LeafSameTypeA() -> ()
    hw.instance "leaf_b" sym @leaf_b @LeafSameTypeB() -> ()
    hw.output
  }

  // CHECK-LABEL: hw.module @TopSameTypeCollision
  // CHECK: %[[TSA:.+]], %[[TSB:.+]] = hw.instance "mid_inst" sym @mid_inst @MidSameTypeCollision() -> (xmr_bored_sig_{{[0-9]+}}: i1, xmr_bored_sig_{{[0-9]+}}: i1)
  // CHECK: hw.output %[[TSA]], %[[TSB]] : i1, i1
  hw.module @TopSameTypeCollision(out out_a : i1, out out_b : i1) {
    hw.instance "mid_inst" sym @mid_inst @MidSameTypeCollision() -> ()
    %ca = sv.xmr.ref @sameTypeCollisionA : !hw.inout<i1>
    %cb = sv.xmr.ref @sameTypeCollisionB : !hw.inout<i1>
    %va = sv.read_inout %ca : !hw.inout<i1>
    %vb = sv.read_inout %cb : !hw.inout<i1>
    hw.output %va, %vb : i1, i1
  }
}
