// RUN: circt-opt --llhd-mem2reg %s | FileCheck %s

// Trivial drive forwarding.
// CHECK-LABEL: @Trivial
hw.module @Trivial(in %u: i42) {
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    %0 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %a, %u after %0 : i42
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42(%u)
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // CHECK: llhd.combinational -> i42
  llhd.combinational -> i42 {
    // CHECK-NOT: llhd.drv
    %0 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %a, %u after %0 : i42
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42(%u)
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: llhd.yield %u
    llhd.yield %1 : i42
  }
}

// Drive forwarding across reconvergent control flow.
// CHECK-LABEL: @ReconvergentControlFlow
hw.module @ReconvergentControlFlow(in %u: i42, in %bool: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : i42
    // CHECK-NEXT: cf.cond_br
    cf.cond_br %bool, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3
  ^bb2:
    cf.br ^bb3
  ^bb3:
    // CHECK: ^bb3:
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42(%u)
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Merging of multiple drives on converging control flow.
// CHECK-LABEL: @DriveMerging
hw.module @DriveMerging(in %u: i42, in %v: i42) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : i42
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42(%u)
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: cf.br ^bb2(%u : i42)
    cf.br ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 : i42
    // CHECK-NOT: llhd.prb
    %2 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42(%v)
    func.call @use_i42(%2) : (i42) -> ()
    // CHECK-NEXT: cf.br ^bb2(%v : i42)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[TMP:%.+]]: i42):
    // CHECK-NOT: llhd.prb
    %3 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42([[TMP]])
    func.call @use_i42(%3) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[TMP]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Forwarding on a subset of control flow when drive dominates all probes.
// CHECK-LABEL: @CompleteDefinitionOnSubset
hw.module @CompleteDefinitionOnSubset(in %u: i42, in %bool: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[UNDEF:%.+]] = hw.constant 0 : i42
    // CHECK-NEXT: [[FALSE:%.+]] = hw.constant false
    // CHECK-NEXT: cf.cond_br %bool, ^bb1, ^bb2([[UNDEF]], [[FALSE]] : i42, i1)
    cf.cond_br %bool, ^bb1, ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT llhd.drv
    llhd.drv %a, %u after %0 : i42
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42(%u)
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: cf.br ^bb2(%u, %true : i42, i1)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[A:%.+]]: i42, [[ACOND:%.+]]: i1):
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[A]] after {{%.+}} if [[ACOND]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Forwarding on a subset of control flow when drive does not dominate all probes.
// CHECK-LABEL: @IncompleteDefinitionOnSubset
hw.module @IncompleteDefinitionOnSubset(in %u: i42, in %bool: i1) {
  // CHECK-NEXT: %true = hw.constant true
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: %false = hw.constant false
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NEXT: cf.cond_br %bool, ^bb1, ^bb2([[A]], %false : i42, i1)
    cf.cond_br %bool, ^bb1, ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT llhd.drv
    llhd.drv %a, %u after %0 : i42
    // CHECK-NEXT: cf.br ^bb2(%u, %true : i42, i1)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[A:%.+]]: i42, [[ACOND:%.+]]: i1):
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42([[A]])
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[A]] after {{%.+}} if [[ACOND]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Check that additional basic blocks get inserted to accommodate probes after
// wait.
// CHECK-LABEL: @InsertProbeBlocks
hw.module @InsertProbeBlocks(in %u: i42) {
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[TMP:%.+]] = llhd.prb %a
    // CHECK-NEXT: cf.br ^bb2([[TMP]] : i42)
    cf.br ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: [[TMP:%.+]] = llhd.prb %a
    // CHECK-NEXT: cf.br ^bb2([[TMP]] : i42)
  ^bb1:
    // CHECK-NEXT: ^bb2([[TMP:%.+]]: i42):
    // CHECK-NOT: llhd.prb
    %0 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42([[TMP]])
    func.call @use_i42(%0) : (i42) -> ()
    // CHECK-NEXT: llhd.wait ^bb1
    llhd.wait ^bb1
  }
}

// Check that no blocks get inserted for definitions that are not driven back to
// their signals.
// CHECK-LABEL: @DontInsertDriveBlocksForProbes
hw.module @DontInsertDriveBlocksForProbes(in %u: i42, in %bool: i1) {
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.prb %a
    llhd.prb %a : i42
    // CHECK-NEXT: cf.cond_br %bool, ^bb1, ^bb3
    cf.cond_br %bool, ^bb1, ^bb3
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: llhd.halt
    llhd.halt
  ^bb2: // no predecessors
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: cf.br ^bb3
    cf.br ^bb3
  ^bb3:
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @MultipleDrivesConverging
hw.module @MultipleDrivesConverging(in %u: i42, in %v: i42) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <i42>
  %b = llhd.sig %v : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : i42
    llhd.drv %b, %v after %0 : i42
    // CHECK-NEXT: cf.br ^bb2(%u, %v : i42, i42)
    cf.br ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 : i42
    llhd.drv %b, %u after %0 : i42
    // CHECK-NEXT: cf.br ^bb2(%v, %u : i42, i42)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[A:%.+]]: i42, [[B:%.+]]: i42):
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42([[A]])
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: cf.br ^bb3
    cf.br ^bb3
  ^bb3:
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[A]]
    // CHECK-NEXT: llhd.drv %b, [[B]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Check that replacing probes with the driven value also updates the probe's
// result value held in the lattice.
// See https://github.com/llvm/circt/issues/8245
// CHECK-LABEL: @ProbeDriveChains
hw.module @ProbeDriveChains() {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  %x = llhd.sig %c0_i42 : <i42>
  %y = llhd.sig %c0_i42 : <i42>
  %z = llhd.sig %c0_i42 : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[TMP:%.+]] = llhd.prb %x
    // CHECK-NOT: llhd.prb
    // CHECK-NOT: llhd.drv
    %1 = llhd.prb %x : i42
    llhd.drv %y, %1 after %0 : i42
    %2 = llhd.prb %y : i42
    llhd.drv %z, %2 after %0 : i42
    %3 = llhd.prb %z : i42
    // CHECK-NEXT: call @use_i42([[TMP]])
    func.call @use_i42(%3) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NOT: llhd.drv %x
    // CHECK-NEXT: llhd.drv %y, [[TMP]]
    // CHECK-NEXT: llhd.drv %z, [[TMP]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Definitions created by inserting initial probes should not generate a drive
// of the probed value back to the signal. Signals driven only in one branch
// should generate conditional drives.
// See https://github.com/llvm/circt/issues/8246
// CHECK-LABEL: @TrackDriveCondition
hw.module @TrackDriveCondition(in %u: i42, in %v: i42) {
  // CHECK: %true = hw.constant true
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : <i42>
  %b = llhd.sig %c0_i42 : <i42>
  %c = llhd.sig %c0_i42 : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[C:%.+]] = llhd.prb %c
    // CHECK-NEXT: %false = hw.constant false
    // CHECK-NEXT: [[B:%.+]] = llhd.prb %b
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : i42
    // CHECK-NEXT: cf.br ^bb2(%u, [[B]], %false, [[C]] : i42, i42, i1, i42)
    cf.br ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: [[C:%.+]] = llhd.prb %c
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 : i42
    llhd.drv %b, %v after %0 : i42
    // CHECK-NEXT: cf.br ^bb2(%v, %v, %true, [[C]] : i42, i42, i1, i42)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[A:%.+]]: i42, [[B:%.+]]: i42, [[BCOND:%.+]]: i1, [[C:%.+]]: i42):
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : i42
    %2 = llhd.prb %b : i42
    %3 = llhd.prb %c : i42
    // CHECK-NEXT: call @use_i42([[A]])
    // CHECK-NEXT: call @use_i42([[B]])
    // CHECK-NEXT: call @use_i42([[C]])
    func.call @use_i42(%1) : (i42) -> ()
    func.call @use_i42(%2) : (i42) -> ()
    func.call @use_i42(%3) : (i42) -> ()
    // CHECK-NEXT: [[T:%.+]] = llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[A]] after [[T]]
    // CHECK-NEXT: llhd.drv %b, [[B]] after [[T]] if [[BCOND]]
    // CHECK-NOT: llhd.drv %c
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  hw.output
}

// Definitions should propagate into loops.
// CHECK-LABEL: @DefinitionsThroughLoops
hw.module @DefinitionsThroughLoops() {
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NEXT: cf.br ^bb1
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.prb
    %0 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42([[A]])
    func.call @use_i42(%0) : (i42) -> ()
    // CHECK-NEXT: cf.br ^bb1
    cf.br ^bb1
  }
}

// Probes should be pulled out of read-modify-write loops, and drives inserted
// when the loop exits.
// CHECK-LABEL: @ReadModifyWriteLoop
hw.module @ReadModifyWriteLoop(in %u: i42) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NEXT: cf.br ^bb1([[A]] : i42)
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: ^bb1([[A:%.+]]: i42):
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: [[ANEW:%.+]] = comb.add [[A]], %u
    %2 = comb.add %1, %u : i42
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %2 after %0 : i42
    // CHECK-NEXT: [[TMP:%.+]] = comb.icmp ult [[A]], %u
    %3 = comb.icmp ult %1, %u : i42
    // CHECK-NEXT: cf.cond_br [[TMP]], ^bb2, ^bb1([[ANEW]] : i42)
    cf.cond_br %3, ^bb2, ^bb1
  ^bb2:
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[ANEW]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// When determining which slots to promote, only uses in the current region
// should be considered.
// CHECK-LABEL: @OnlyConsiderUsesInRegionForPromotability
hw.module @OnlyConsiderUsesInRegionForPromotability(in %u: i42) {
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    func.call @use_ref_i42(%a) : (!llhd.ref<i42>) -> ()
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    // CHECK-NOT: llhd.prb
    %0 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %a, %u after %0 : i42
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42(%u)
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Probes that are live across wait ops must be captured as destination operands
// of the wait op to allow drives to be forwarded to the probes.
// CHECK-LABEL: @CaptureAcrossWaits
hw.module @CaptureAcrossWaits(in %u: i42, in %bool: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  %a = llhd.sig %c0_i42 : <i42>
  %b = llhd.sig %c0_i42 : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: cf.cond_br %bool, ^bb1, ^bb5([[A]] : i42)
    cf.cond_br %bool, ^bb1, ^bb5
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: llhd.wait ^bb2([[A]] : i42)
    llhd.wait ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[A:%.+]]: i42):
    // CHECK-NEXT: cf.cond_br %bool, ^bb3, ^bb6([[A]] : i42)
    cf.cond_br %bool, ^bb3, ^bb6
  ^bb3:
    // CHECK-NEXT: ^bb3
    // CHECK-NEXT: llhd.wait ^bb4([[A]] : i42)
    llhd.wait ^bb4
  ^bb4:
    // CHECK-NEXT: ^bb4([[A:%.+]]: i42):
    // CHECK-NEXT: cf.br ^bb5([[A]] : i42)
    cf.br ^bb5
  ^bb5:
    // CHECK-NEXT: ^bb5([[A:%.+]]: i42):
    // CHECK-NEXT: cf.br ^bb6([[A]] : i42)
    cf.br ^bb6
  ^bb6:
    // CHECK-NEXT: ^bb6([[A:%.+]]: i42):
    // CHECK-NEXT: call @use_i42([[A]])
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Conditional drive forwarding.
// CHECK-LABEL: @ConditionalDrives
hw.module @ConditionalDrives(in %u: i42, in %v: i42, in %q: i1, in %r: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u after {{%.+}} if %q
    llhd.drv %a, %u after %0 if %q : i42
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 if %q : i42
    // CHECK-NEXT: cf.br ^bb2(%u : i42)
    cf.br ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 if %q : i42
    // CHECK-NEXT: cf.br ^bb2(%v : i42)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[A:%.+]]: i42):
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[A]] after {{%.+}} if %q
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 if %q : i42
    // CHECK-NEXT: cf.br ^bb2(%u, %q : i42, i1)
    cf.br ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 if %r : i42
    // CHECK-NEXT: cf.br ^bb2(%v, %r : i42, i1)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[A:%.+]]: i42, [[ACOND:%.+]]: i1):
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[A]] after {{%.+}} if [[ACOND]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// See https://github.com/llvm/circt/issues/8494.
// CHECK-LABEL: @MultipleConditionalDrives
hw.module @MultipleConditionalDrives(in %u: i42, in %v: i42, in %w: i42, in %q: i1, in %r: i1, in %s: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <i42>
  // Conditional drives following non-conditional drives should create
  // multiplexers to modify the value forwarded as a reaching definition.
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : i42
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %q, %v, %u : i42
    llhd.drv %a, %v after %0 if %q : i42
    // CHECK-NEXT: [[DRV2:%.+]] = comb.mux %r, %w, [[DRV1]] : i42
    llhd.drv %a, %w after %0 if %r : i42
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[DRV2]] after {{%.+}} :
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // Subsequent conditional drives should create multiplexers to combine the
  // different possible drive values, and they should aggregate drive conditions
  // with OR gates.
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 if %q : i42
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %r, %v, %u : i42
    // CHECK-NEXT: [[ENABLE1:%.+]] = comb.or %r, %q : i1
    llhd.drv %a, %v after %0 if %r : i42
    // CHECK-NEXT: [[DRV2:%.+]] = comb.mux %s, %w, [[DRV1]] : i42
    // CHECK-NEXT: [[ENABLE2:%.+]] = comb.or %s, [[ENABLE1]] : i1
    llhd.drv %a, %w after %0 if %s : i42
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[DRV2]] after {{%.+}} if [[ENABLE2]] :
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // Probe after chain of conditional drives.
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %q, %u, [[A]] : i42
    llhd.drv %a, %u after %0 if %q : i42
    // CHECK-NEXT: [[DRV2:%.+]] = comb.mux %r, %v, [[DRV1]] : i42
    // CHECK-NEXT: [[ENABLE2:%.+]] = comb.or %r, %q : i1
    llhd.drv %a, %v after %0 if %r : i42
    // CHECK-NEXT: [[DRV3:%.+]] = comb.mux %s, %w, [[DRV2]] : i42
    // CHECK-NEXT: [[ENABLE3:%.+]] = comb.or %s, [[ENABLE2]] : i1
    llhd.drv %a, %w after %0 if %s : i42
    // CHECK-NEXT: call @use_i42([[DRV3]])
    %1 = llhd.prb %a : i42
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[DRV3]] after {{%.+}} if [[ENABLE3]] :
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // Probe after unconditional drive followed by chain of conditional drives.
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : i42
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %q, %v, %u : i42
    llhd.drv %a, %v after %0 if %q : i42
    // CHECK-NEXT: [[DRV2:%.+]] = comb.mux %r, %w, [[DRV1]] : i42
    llhd.drv %a, %w after %0 if %r : i42
    // CHECK-NEXT: call @use_i42([[DRV2]])
    %1 = llhd.prb %a : i42
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[DRV2]] after {{%.+}} :
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Delayed and blocking drive interaction.
// CHECK-LABEL: @DelayedDrives
hw.module @DelayedDrives(in %u: i42, in %v: i42, in %bool: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = llhd.constant_time <0ns, 1d, 0e>
  %a = llhd.sig %u : <i42>
  // Delayed drives after blocking drives persist.
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[T:%.+]] = llhd.constant_time <0ns, 0d, 1e>
    // CHECK-NEXT: llhd.drv %a, %u after [[T]]
    // CHECK-NEXT: [[T:%.+]] = llhd.constant_time <0ns, 1d, 0e>
    // CHECK-NEXT: llhd.drv %a, %v after [[T]]
    llhd.drv %a, %u after %0 : i42
    llhd.drv %a, %v after %1 : i42
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // Later blocking drives erase earlier delayed drives.
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv %a, %u
    // CHECK-NEXT: [[T:%.+]] = llhd.constant_time <0ns, 0d, 1e>
    // CHECK-NEXT: llhd.drv %a, %v after [[T]]
    llhd.drv %a, %u after %1 : i42
    llhd.drv %a, %v after %0 : i42
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %1 : i42
    // CHECK-NEXT: hw.constant 0 : i42
    // CHECK-NEXT: hw.constant false
    // CHECK-NEXT: cf.cond_br %bool, ^bb1, ^bb2({{%c0_i42.*}}, {{%false.*}}, %u, {{%true.*}} : i42, i1, i42, i1)
    cf.cond_br %bool, ^bb1, ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 : i42
    // CHECK-NEXT: hw.constant 0 : i42
    // CHECK-NEXT: hw.constant false
    // CHECK-NEXT: cf.br ^bb2(%v, {{%true.*}}, {{%c0_i42.*}}, {{%false.*}} : i42, i1, i42, i1)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[ABLK:%.+]]: i42, [[ABLKCOND:%.+]]: i1, [[ADEL:%.+]]: i42, [[ADELCOND:%.+]]: i1):
    // CHECK-NEXT: [[T:%.+]] = llhd.constant_time <0ns, 0d, 1e>
    // CHECK-NEXT: llhd.drv %a, [[ABLK]] after [[T]] if [[ABLKCOND]]
    // CHECK-NEXT: [[T:%.+]] = llhd.constant_time <0ns, 1d, 0e>
    // CHECK-NEXT: llhd.drv %a, [[ADEL]] after [[T]] if [[ADELCOND]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @DelayedConditionalDrives
hw.module @DelayedConditionalDrives(in %u: i42, in %v: i42, in %w: i42, in %q: i1, in %r: i1, in %s: i1) {
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : i42
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %q, %v, %u : i42
    llhd.drv %a, %v after %0 if %q : i42
    // CHECK-NEXT: [[DRV2:%.+]] = comb.mux %r, %w, [[DRV1]] : i42
    llhd.drv %a, %w after %0 if %r : i42
    // CHECK-NEXT: llhd.constant_time <0ns, 1d, 0e>
    // CHECK-NEXT: llhd.drv %a, [[DRV2]] after {{%.+}} :
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Basic probing of signal projection works.
// CHECK-LABEL: @BasicProjectionProbe
hw.module @BasicProjectionProbe(in %u: !hw.array<4xi42>, in %v: i42, in %i: i2) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <!hw.array<4xi42>>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.sig.array_get
    // CHECK-NOT: llhd.drv
    %1 = llhd.sig.array_get %a[%i] : <!hw.array<4xi42>>
    llhd.drv %a, %u after %0 : !hw.array<4xi42>
    // CHECK-NOT: llhd.prb
    // CHECK-NEXT: [[TMP:%.+]] = hw.array_get %u[%i]
    %2 = llhd.prb %1 : i42
    // CHECK-NEXT: call @use_i42([[TMP]])
    func.call @use_i42(%2) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Basic driving of signal projection works.
// CHECK-LABEL: @BasicProjectionDrive
hw.module @BasicProjectionDrive(in %u: !hw.array<4xi42>, in %v: i42, in %i: i2) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <!hw.array<4xi42>>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.array<4xi42>
    // CHECK-NOT: llhd.sig.array_get
    %1 = llhd.sig.array_get %a[%i] : <!hw.array<4xi42>>
    // CHECK-NOT: llhd.drv
    // CHECK-NEXT: [[A:%.+]] = hw.array_inject %u[%i], %v
    llhd.drv %1, %v after %0 : i42
    // CHECK-NOT: llhd.prb
    %2 = llhd.prb %a : !hw.array<4xi42>
    // CHECK-NEXT: call @use_array_i42([[A]])
    func.call @use_array_i42(%2) : (!hw.array<4xi42>) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[A]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Conditional drives of signal projections.
// CHECK-LABEL: @ConditionalProjectionDrive
hw.module @ConditionalProjectionDrive(in %u: !hw.array<4xi42>, in %v: i42, in %w: i42, in %i: i2, in %q: i1, in %r: i1, in %s: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <!hw.array<4xi42>>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.sig.array_get
    %1 = llhd.sig.array_get %a[%i] : <!hw.array<4xi42>>
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.array<4xi42>
    // CHECK-NEXT: [[TMP:%.+]] = hw.array_get %u[%i]
    // CHECK-NEXT: [[FIELD1:%.+]] = comb.mux %q, %v, [[TMP]]
    // CHECK-NEXT: [[DRV1:%.+]] = hw.array_inject %u[%i], [[FIELD1]]
    llhd.drv %1, %v after %0 if %q : i42
    // CHECK-NEXT: [[FIELD2:%.+]] = comb.mux %r, %w, [[FIELD1]]
    // CHECK-NEXT: [[DRV2:%.+]] = hw.array_inject [[DRV1]][%i], [[FIELD2]]
    llhd.drv %1, %w after %0 if %r : i42
    // CHECK-NEXT: call @use_array_i42([[DRV2]])
    // CHECK-NEXT: call @use_i42([[FIELD2]])
    %2 = llhd.prb %a : !hw.array<4xi42>
    %3 = llhd.prb %1 : i42
    func.call @use_array_i42(%2) : (!hw.array<4xi42>) -> ()
    func.call @use_i42(%3) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[DRV2]] after {{%.+}} :
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NOT: llhd.sig.array_get
    %1 = llhd.sig.array_get %a[%i] : <!hw.array<4xi42>>
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %q, %u, [[A]]
    llhd.drv %a, %u after %0 if %q : !hw.array<4xi42>
    // CHECK-NEXT: [[TMP:%.+]] = hw.array_get [[DRV1]][%i]
    // CHECK-NEXT: [[FIELD2:%.+]] = comb.mux %r, %v, [[TMP]]
    // CHECK-NEXT: [[DRV2:%.+]] = hw.array_inject [[DRV1]][%i], [[FIELD2]]
    // CHECK-NEXT: [[ENABLE2:%.+]] = comb.or %r, %q
    llhd.drv %1, %v after %0 if %r : i42
    // CHECK-NEXT: [[FIELD3:%.+]] = comb.mux %s, %w, [[FIELD2]]
    // CHECK-NEXT: [[DRV3:%.+]] = hw.array_inject [[DRV2]][%i], [[FIELD3]]
    // CHECK-NEXT: [[ENABLE3:%.+]] = comb.or %s, [[ENABLE2]]
    llhd.drv %1, %w after %0 if %s : i42
    // CHECK-NEXT: call @use_array_i42([[DRV3]])
    // CHECK-NEXT: call @use_i42([[FIELD3]])
    %2 = llhd.prb %a : !hw.array<4xi42>
    %3 = llhd.prb %1 : i42
    func.call @use_array_i42(%2) : (!hw.array<4xi42>) -> ()
    func.call @use_i42(%3) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[DRV3]] after {{%.+}} if [[ENABLE3]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Delayed drives of signal projections should fall back to probing the current
// value of the entire slot and injecting into that. This is equivalent to
// treating all arrays as packed.
// CHECK-LABEL: @DelayedProjectionDrive
hw.module @DelayedProjectionDrive(in %u: !hw.array<4xi42>, in %v: i42, in %i: i2) {
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %a = llhd.sig %u : <!hw.array<4xi42>>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    %1 = llhd.sig.array_get %a[%i] : <!hw.array<4xi42>>
    // CHECK-NEXT: [[TMP:%.+]] = hw.array_inject [[A]][%i], %v
    llhd.drv %1, %v after %0 : i42
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[TMP]] after {{%.+}}
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @ProjectionThroughBlockArg
hw.module @ProjectionThroughBlockArg(in %u: !hw.array<4xi42>, in %v: i42, in %i: i2) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <!hw.array<4xi42>>
  // CHECK: llhd.process
  llhd.process {
    %1 = llhd.sig.array_get %a[%i] : <!hw.array<4xi42>>
    // Pass projection through block argument
    cf.br ^bb1(%1 : !llhd.ref<i42>)
  ^bb1(%2: !llhd.ref<i42>):
    // CHECK: ^bb1([[ARG:%.+]]: !llhd.ref<i42>):
    // CHECK-NEXT: llhd.prb [[ARG]]
    %3 = llhd.prb %2 : i42
    // CHECK-NEXT: llhd.drv [[ARG]]
    llhd.drv %2, %v after %0 : i42
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @MultipleArrayGetsSameIndex
hw.module @MultipleArrayGetsSameIndex(in %u: !hw.array<4xi42>, in %v: i42, in %w: i42, in %i: i2) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <!hw.array<4xi42>>
  // CHECK: llhd.process
  llhd.process {
    // Two separate array_gets for the same index
    // CHECK-NOT: llhd.sig.array_get
    %get1 = llhd.sig.array_get %a[%i] : <!hw.array<4xi42>>
    %get2 = llhd.sig.array_get %a[%i] : <!hw.array<4xi42>>
    // Drive both projections with different values
    // CHECK-NOT: llhd.drv
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NEXT: [[DRV1:%.+]] = hw.array_inject [[A]][%i], %v
    // CHECK-NEXT: [[DRV2:%.+]] = hw.array_inject [[DRV1]][%i], %w
    llhd.drv %get1, %v after %0 : i42
    llhd.drv %get2, %w after %0 : i42
    // Probe both projections
    // CHECK-NOT: llhd.prb
    %prb1 = llhd.prb %get1 : i42
    %prb2 = llhd.prb %get2 : i42
    // CHECK-NEXT: call @use_i42(%w)
    // CHECK-NEXT: call @use_i42(%w)
    func.call @use_i42(%prb1) : (i42) -> ()
    func.call @use_i42(%prb2) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[DRV2]] after {{%.+}}
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @NestedArrayGet3D
hw.module @NestedArrayGet3D(
  in %u: !hw.array<5xarray<6xarray<7xi42>>>,
  in %v: i42, in %i: i3, in %j: i3, in %k: i3
) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <!hw.array<5xarray<6xarray<7xi42>>>>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // Three nested projections
    %get1 = llhd.sig.array_get %a[%i] : <!hw.array<5xarray<6xarray<7xi42>>>>
    %get2 = llhd.sig.array_get %get1[%j] : <!hw.array<6xarray<7xi42>>>
    %get3 = llhd.sig.array_get %get2[%k] : <!hw.array<7xi42>>
    // Drive the innermost projection
    // CHECK-NEXT: [[GET3:%.+]] = hw.array_get [[A]][%i]
    // CHECK-NEXT: [[GET2:%.+]] = hw.array_get [[GET3]][%j]
    // CHECK-NEXT: [[INJECT1:%.+]] = hw.array_inject [[GET2]][%k], %v
    // CHECK-NEXT: [[INJECT2:%.+]] = hw.array_inject [[GET3]][%j], [[INJECT1]]
    // CHECK-NEXT: [[INJECT3:%.+]] = hw.array_inject [[A]][%i], [[INJECT2]]
    llhd.drv %get3, %v after %0 : i42
    // Probe the innermost projection
    %prb = llhd.prb %get3 : i42
    // CHECK-NEXT: call @use_i42(%v)
    func.call @use_i42(%prb) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[INJECT3]] after {{%.+}}
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @BasicSigExtract
hw.module @BasicSigExtract(in %u: i42, in %v: i10, in %i: i6, in %q: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : i42
    // CHECK-NOT: llhd.sig.extract
    %1 = llhd.sig.extract %a from %i : <i42> -> <i10>
    // CHECK-NOT: llhd.drv
    // CHECK-NEXT: [[EXT1:%.+]] = hw.constant 0 : i36
    // CHECK-NEXT: [[EXT2:%.+]] = comb.concat [[EXT1]], %i : i36, i6
    // CHECK-NEXT: [[EXT3:%.+]] = comb.shru %u, [[EXT2]] : i42
    // CHECK-NEXT: [[EXT4:%.+]] = comb.extract [[EXT3]] from 0 : (i42) -> i10
    // CHECK-NEXT: [[MUX:%.+]] = comb.mux %q, %v, [[EXT4]] : i10
    // CHECK-NEXT: [[INJ1:%.+]] = hw.constant 0 : i36
    // CHECK-NEXT: [[INJ2:%.+]] = comb.concat [[INJ1]], %i : i36, i6
    // CHECK-NEXT: [[INJ3:%.+]] = hw.constant 1023 : i42
    // CHECK-NEXT: [[INJ4:%.+]] = comb.shl [[INJ3]], [[INJ2]] : i42
    // CHECK-NEXT: [[INJ5:%.+]] = hw.constant -1 : i42
    // CHECK-NEXT: [[INJ6:%.+]] = comb.xor bin [[INJ4]], [[INJ5]] : i42
    // CHECK-NEXT: [[INJ7:%.+]] = comb.and %u, [[INJ6]] : i42
    // CHECK-NEXT: [[INJ8:%.+]] = hw.constant 0 : i32
    // CHECK-NEXT: [[INJ9:%.+]] = comb.concat [[INJ8]], [[MUX]] : i32, i10
    // CHECK-NEXT: [[INJ10:%.+]] = comb.shl [[INJ9]], [[INJ2]] : i42
    // CHECK-NEXT: [[INJ11:%.+]] = comb.or [[INJ7]], [[INJ10]] : i42
    llhd.drv %1, %v after %0 if %q : i10
    // CHECK-NOT: llhd.prb
    %2 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42([[INJ11]])
    func.call @use_i42(%2) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[INJ11]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @BasicStructExtract
hw.module @BasicStructExtract(in %u: !hw.struct<f: i42>, in %v: i42, in %q: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <!hw.struct<f: i42>>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.struct<f: i42>
    // CHECK-NOT: llhd.sig.struct_extract
    %1 = llhd.sig.struct_extract %a["f"] : <!hw.struct<f: i42>>
    // CHECK-NOT: llhd.drv
    // CHECK-NEXT: [[EXT:%.+]] = hw.struct_extract %u["f"]
    // CHECK-NEXT: [[MUX:%.+]] = comb.mux %q, %v, [[EXT]]
    // CHECK-NEXT: [[INJ:%.+]] = hw.struct_inject %u["f"], [[MUX]]
    llhd.drv %1, %v after %0 if %q : i42
    // CHECK-NOT: llhd.prb
    %2 = llhd.prb %1 : i42
    // CHECK-NEXT: call @use_i42([[MUX]])
    func.call @use_i42(%2) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[INJ]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @CombCreateDynamicInject
hw.module @CombCreateDynamicInject(in %u: i42, in %v: i10, in %q: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <i42>

  // offset = 0
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[TMP1:%.+]] = comb.extract %u from 10 : (i42) -> i32
    // CHECK-NEXT: [[TMP2:%.+]] = comb.concat [[TMP1]], %v : i32, i10
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[TMP2]]
    // CHECK-NEXT: llhd.halt
    %c0_i6 = hw.constant 0 : i6
    %1 = llhd.sig.extract %a from %c0_i6 : <i42> -> <i10>
    llhd.drv %a, %u after %0 : i42
    llhd.drv %1, %v after %0 : i10
    llhd.halt
  }

  // offset > 0, end < 42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[TMP1:%.+]] = comb.extract %u from 30 : (i42) -> i12
    // CHECK-NEXT: [[TMP2:%.+]] = comb.extract %u from 0 : (i42) -> i20
    // CHECK-NEXT: [[TMP3:%.+]] = comb.concat [[TMP1]], %v, [[TMP2]] : i12, i10, i20
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[TMP3]]
    // CHECK-NEXT: llhd.halt
    %c20_i6 = hw.constant 20 : i6
    %1 = llhd.sig.extract %a from %c20_i6 : <i42> -> <i10>
    llhd.drv %a, %u after %0 : i42
    llhd.drv %1, %v after %0 : i10
    llhd.halt
  }

  // end = 42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[TMP1:%.+]] = comb.extract %u from 0 : (i42) -> i32
    // CHECK-NEXT: [[TMP2:%.+]] = comb.concat %v, [[TMP1]] : i10, i32
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[TMP2]]
    // CHECK-NEXT: llhd.halt
    %c32_i6 = hw.constant 32 : i6
    %1 = llhd.sig.extract %a from %c32_i6 : <i42> -> <i10>
    llhd.drv %a, %u after %0 : i42
    llhd.drv %1, %v after %0 : i10
    llhd.halt
  }

  // offset < 42, end > 42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[TMP1:%.+]] = comb.extract %v from 0 : (i10) -> i5
    // CHECK-NEXT: [[TMP2:%.+]] = comb.extract %u from 0 : (i42) -> i37
    // CHECK-NEXT: [[TMP3:%.+]] = comb.concat [[TMP1]], [[TMP2]] : i5, i37
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[TMP3]]
    // CHECK-NEXT: llhd.halt
    %c37_i6 = hw.constant 37 : i6
    %1 = llhd.sig.extract %a from %c37_i6 : <i42> -> <i10>
    llhd.drv %a, %u after %0 : i42
    llhd.drv %1, %v after %0 : i10
    llhd.halt
  }

  // offset >= 42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: llhd.halt
    %c42_i6 = hw.constant 42 : i6
    %1 = llhd.sig.extract %a from %c42_i6 : <i42> -> <i10>
    llhd.drv %a, %u after %0 : i42
    llhd.drv %1, %v after %0 : i10
    llhd.halt
  }
}

func.func private @use_i8(%arg0: i8)
func.func private @use_i32(%arg0: i32)
func.func private @use_i42(%arg0: i42)
func.func private @use_ref_i42(%arg0: !llhd.ref<i42>)
func.func private @use_array_i42(%arg0: !hw.array<4xi42>)
func.func private @use_array_i8(%arg0: !hw.array<4xi8>)
func.func private @use_union(%arg0: !hw.union<a: i8, b: i8>)

// Regression test that verifies probe is inserted post use.
// CHECK-LABEL: ProbePostDef
hw.module @ProbePostDef() {
  %2 = llhd.combinational -> i1 {
    %false = hw.constant false
    %e = llhd.sig %false : <i1>
    %4 = llhd.prb %e : i1
    llhd.yield %4 : i1
  }
  hw.output
}

// CHECK-LABEL: DominanceTest1
// Regression test verifying that signal definitions do not propagate to blocks
// that are not dominated by the signal. Otherwise, the inserted drive would not
// dominate signal operand.
hw.module @DominanceTest1() {
  %true = hw.constant true
  %false = hw.constant false
  %3 = llhd.constant_time <0ns, 0d, 1e>
  %clock_0 = llhd.sig name "clock" %false : <i1>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[TMP0:%.+]] = llhd.prb %clock
    // CHECK-NEXT: cf.br ^bb1([[TMP0]] : i1)
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: ^bb1([[BBARG0:%.+]]: i1)
    // CHECK-NEXT: llhd.wait ([[BBARG0]] : i1), ^bb2
    %5 = llhd.prb %clock_0 : i1
    llhd.wait (%5 : i1), ^bb3
  ^bb3:
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: [[TMP1:%.+]] = llhd.prb %clock
    // CHECK-NEXT: cf.br ^bb1([[TMP1]] : i1)
    %c0_i4 = hw.constant 0 : i4
    %ready_T = llhd.sig %c0_i4 : <i4>
    llhd.drv %ready_T, %c0_i4 after %3 : i4
    %6 = llhd.prb %ready_T : i4
    cf.br ^bb1
  }
  hw.output
}

// CHECK-LABEL: DominanceTest2
hw.module @DominanceTest2() {
  %false = hw.constant false
  %b = llhd.sig %false : <i1>
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK-NEXT: cf.br ^bb3
    cf.br ^bb3
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: cf.br ^bb2
    %0 = llhd.prb %b : i1
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %g = llhd.sig %false : <i1>
    llhd.drv %g, %0 after %1 : i1
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2
    // CHECK-NEXT: cf.br ^bb3
    cf.br ^bb3
  ^bb3:
    llhd.yield
  }
  hw.output
}

// CHECK-LABEL: @ProjectionAndDriveInDifferentBlocks
hw.module @ProjectionAndDriveInDifferentBlocks(in %u: !hw.struct<f: i42>, in %v: i42, in %q: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <!hw.struct<f: i42>>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.struct<f: i42>
    // CHECK-NOT: llhd.sig.struct_extract
    %1 = llhd.sig.struct_extract %a["f"] : <!hw.struct<f: i42>>
    // CHECK-NEXT: cf.br ^bb1
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: ^bb1
    // CHECK-NOT: llhd.drv
    // CHECK-NEXT: [[EXT:%.+]] = hw.struct_extract %u["f"]
    // CHECK-NEXT: [[MUX:%.+]] = comb.mux %q, %v, [[EXT]]
    // CHECK-NEXT: [[INJ:%.+]] = hw.struct_inject %u["f"], [[MUX]]
    llhd.drv %1, %v after %0 if %q : i42
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[INJ]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Local signals should be removed.
// CHECK-LABEL: @LocalSignals
hw.module @LocalSignals(in %u: i42, in %v: i42) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.sig
    %a = llhd.sig %u : <i42>
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42(%u)
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 : i42
    // CHECK-NOT: llhd.prb
    %2 = llhd.prb %a : i42
    // CHECK-NEXT: call @use_i42(%v)
    func.call @use_i42(%2) : (i42) -> ()
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Make sure all values live across wait terminators are captured as block
// arguments on the wait. This allows us to more generally handle projections
// into probes, and values computed from probes.
// See https://github.com/llvm/circt/pull/9481
// CHECK-LABEL: @CaptureNonProbeValuesAcrossWait
hw.module @CaptureNonProbeValuesAcrossWait(in %a: i42, in %b: i42) {
  llhd.process {
    // CHECK: [[TMP1:%.+]] = comb.and %b, %b
    %0 = comb.and %b, %b : i42
    // CHECK: llhd.wait (%a : i42), ^bb1([[TMP1]] : i42)
    llhd.wait (%a : i42), ^bb1
    // CHECK: ^bb1([[TMP2:%.+]]: i42):
  ^bb1:
    // CHECK: comb.or [[TMP2]], [[TMP2]]
    comb.or %0, %0 : i42
    llhd.halt
  }
}

// Values defined after a wait should not be captured across that wait. They may
// still need to be captured across later waits that their definition dominates.
// CHECK-LABEL: @CapturePostWaitValueOnlyAcrossDominatedWaits
hw.module @CapturePostWaitValueOnlyAcrossDominatedWaits(in %a: i42) {
  %d = llhd.constant_time <1fs, 0d, 0e>
  llhd.process {
    // CHECK: llhd.wait delay [[D:%.+]], ^bb1
    llhd.wait delay %d, ^bb1
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: [[VALUE:%.+]] = comb.xor %a, %a
    %v = comb.xor %a, %a : i42
    // CHECK-NEXT: llhd.wait delay [[D]], ^bb2([[VALUE]] : i42)
    llhd.wait delay %d, ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[CAPTURED:%.+]]: i42):
    // CHECK-NEXT: call @use_i42([[CAPTURED]])
    func.call @use_i42(%v) : (i42) -> ()
    llhd.halt
  }
}

// Float signals are promotable; Mem2Reg materializes a floating-point zero
// default for the merge and forwards the driven value through block arguments,
// turning the drive into a single conditional drive. Extremely wide aggregate
// signals still are not promotable since default values would exceed MLIR's
// IntegerType limit.
// CHECK-LABEL: @RealSignalDrivePromoted
hw.module @RealSignalDrivePromoted(in %clk : i1, in %a : f64, in %b : f64) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: %r = llhd.sig %b : <f64>
  %r = llhd.sig %b : <f64>
  // CHECK: llhd.process
  // CHECK: arith.constant 0.000000e+00 : f64
  // CHECK: ^bb1([[VAL:%.+]]: f64, [[EN:%.+]]: i1):
  // CHECK:   llhd.drv %r, [[VAL]] after {{%.+}} if [[EN]] : f64
  // CHECK:   llhd.wait (%clk : i1), ^bb2
  // CHECK: ^bb2:
  // CHECK:   cf.br ^bb1(%a, %true : f64, i1)
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait (%clk : i1), ^bb2
  ^bb2:
    llhd.drv %r, %a after %0 : f64
    cf.br ^bb1
  }
}

// CHECK-LABEL: @TooWideSignalNotPromoted
hw.module @TooWideSignalNotPromoted(
  in %clk : i1,
  in %a : !hw.array<2097153xi8>,
  in %b : !hw.array<2097153xi8>
) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: %r = llhd.sig %b : <!hw.array<2097153xi8>>
  %r = llhd.sig %b : <!hw.array<2097153xi8>>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait (%clk : i1), ^bb2
  ^bb2:
    // CHECK: llhd.drv %r, %a after
    llhd.drv %r, %a after %0 : !hw.array<2097153xi8>
    cf.br ^bb1
  }
}

// Promote a signal of union type. The SigStructExtractOp is used for both
// struct and union member access; Mem2Reg must use the appropriate HW ops
// for unpacking (union_extract) and packing (union_create).
// CHECK-LABEL: @UnionSignalPromoted
hw.module @UnionSignalPromoted(in %u : !hw.union<a: i8, b: i8>, in %v : i8, in %q : i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0 = hw.constant 0 : i8
  %init = hw.bitcast %c0 : (i8) -> !hw.union<a: i8, b: i8>
  %a = llhd.sig %init : <!hw.union<a: i8, b: i8>>
  // CHECK: llhd.process
  llhd.process {
    // Drive the whole union, then project a field.
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.union<a: i8, b: i8>
    // CHECK-NOT: llhd.sig.struct_extract
    %1 = llhd.sig.struct_extract %a["a"] : <!hw.union<a: i8, b: i8>>
    // Conditionally drive the field (tests unpack and pack with unions).
    // CHECK-NOT: llhd.drv
    // CHECK-NEXT: [[EXT:%.+]] = hw.union_extract %u["a"]
    // CHECK-NEXT: [[MUX:%.+]] = comb.mux %q, %v, [[EXT]]
    // CHECK-NEXT: [[INJ:%.+]] = hw.union_create "a", [[MUX]]
    llhd.drv %1, %v after %0 if %q : i8
    // Probe the field (the value should be forwarded through the union).
    // CHECK-NOT: llhd.prb
    // CHECK-NEXT: [[READFIELD:%.+]] = hw.union_extract [[INJ]]["a"]
    %2 = llhd.prb %1 : i8
    // CHECK-NEXT: call @use_i8([[READFIELD]])
    func.call @use_i8(%2) : (i8) -> ()
    // Probe the whole union (should see the union_create result).
    // CHECK-NOT: llhd.prb
    %3 = llhd.prb %a : !hw.union<a: i8, b: i8>
    // CHECK-NEXT: call @use_union([[INJ]])
    func.call @use_union(%3) : (!hw.union<a: i8, b: i8>) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[INJ]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Don't promote a signal if a projection op (like sig.array_get) has a nested
// projection user (like sig.extract) in a different block. Mem2Reg rewrites
// signal references across block boundaries and would break the projection
// chain, causing getProjections to encounter a BlockArgument.
// CHECK-LABEL: @NestedProjectionAcrossBlocks
hw.module @NestedProjectionAcrossBlocks(in %v : i4) {
  %t = llhd.constant_time <0ns, 0d, 1e>
  %d = llhd.constant_time <1ns, 0d, 0e>
  %true = hw.constant true
  %c0 = hw.constant 0 : i8
  %c4 = hw.constant -4 : i3
  %init = hw.aggregate_constant [0 : i8, 0 : i8] : !hw.array<2xi8>
  // CHECK: %mem = llhd.sig
  %mem = llhd.sig %init : <!hw.array<2xi8>>
  llhd.process {
    %e = llhd.sig.array_get %mem[%true] : <!hw.array<2xi8>>
    llhd.drv %e, %c0 after %t : i8
    llhd.wait delay %d, ^bb1
  ^bb1:
    // This sig.extract is a nested projection of %e, but in a different block.
    // CHECK: llhd.sig.extract
    %sub = llhd.sig.extract %e from %c4 : <i8> -> <i4>
    llhd.drv %sub, %v after %t : i4
    llhd.halt
  }
}

// Don't promote a signal if a projection has multiple drives with different
// delays. Mem2Reg splits blocking and delta drives into separate slot tracking,
// but the reaching definition analysis doesn't handle this correctly for
// projections, causing a "no definition reaches drive" assertion.
// CHECK-LABEL: @MultiDelayProjectionDrive
hw.module @MultiDelayProjectionDrive() {
  %t_eps = llhd.constant_time <0ns, 0d, 1e>
  %t_delta = llhd.constant_time <0ns, 1d, 0e>
  %c0 = hw.constant 0 : i8
  %true = hw.constant true
  %init = hw.aggregate_constant [0 : i8, 0 : i8] : !hw.array<2xi8>
  // CHECK: %sig = llhd.sig
  %sig = llhd.sig %init : <!hw.array<2xi8>>
  llhd.process {
    %e = llhd.sig.array_get %sig[%true] : <!hw.array<2xi8>>
    // CHECK: llhd.drv
    llhd.drv %e, %c0 after %t_eps : i8
    // CHECK: llhd.drv
    llhd.drv %e, %c0 after %t_delta : i8
    llhd.halt
  }
}

// Same-successor conditional branches appear as duplicate predecessor edges in
// the CFG. Mem2Reg must update every edge in the terminator, but only once per
// predecessor terminator, so it does not append duplicate operands without
// matching block arguments.
// CHECK-LABEL: @SameSuccessorCondBr
hw.module @SameSuccessorCondBr(in %clk : i1) {
  %t_delta = llhd.constant_time <0ns, 1d, 0e>
  %t_eps = llhd.constant_time <0ns, 0d, 1e>
  %true = hw.constant true
  %false = hw.constant false
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c5_i32 = hw.constant 5 : i32
  %cnt = llhd.sig %c0_i32 : <i32>
  %proc:2 = llhd.process -> i1, i1 {
    cf.br ^bb1(%clk, %c0_i32, %false, %c0_i32, %false, %false, %false : i1, i32, i1, i32, i1, i1, i1)
  // CHECK: ^bb1({{.*}}: i1, {{.*}}: i32, {{.*}}: i1, {{.*}}: i32, {{.*}}: i1, {{.*}}: i1, {{.*}}: i1, {{.*}}: i32):
  ^bb1(%0: i1, %1: i32, %2: i1, %3: i32, %4: i1, %5: i1, %6: i1):
    llhd.drv %cnt, %1 after %t_eps if %2 : i32
    llhd.drv %cnt, %3 after %t_delta if %4 : i32
    llhd.wait yield (%5, %6 : i1, i1), (%clk : i1), ^bb2(%0 : i1)
  ^bb2(%7: i1):
    %8 = llhd.prb %cnt : i32
    %9 = comb.xor bin %7, %true : i1
    %10 = comb.and bin %9, %clk : i1
    cf.cond_br %10, ^bb3, ^bb1(%clk, %8, %false, %8, %false, %false, %false : i1, i32, i1, i32, i1, i1, i1)
  ^bb3:
    %11 = comb.add %8, %c1_i32 : i32
    %12 = comb.icmp sgt %8, %c5_i32 : i32
    // CHECK: cf.cond_br {{.*}}, ^bb1({{.*}} : i1, i32, i1, i32, i1, i1, i1, i32), ^bb1({{.*}} : i1, i32, i1, i32, i1, i1, i1, i32)
    cf.cond_br %12, ^bb1(%clk, %c5_i32, %true, %c0_i32, %false, %true, %true : i1, i32, i1, i32, i1, i1, i1), ^bb1(%clk, %8, %false, %11, %true, %false, %false : i1, i32, i1, i32, i1, i1, i1)
  }
}

// Test that the following doesn't hang.
//
// See: https://github.com/llvm/circt/issues/10314
//
// CHECK-LABEL: @Timeout_10314
hw.module private @Timeout_10314() {
  %c0_i5 = hw.constant 0 : i5
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i32 = hw.constant 0 : i32
  %c0_i4 = hw.constant 0 : i4
  %c0_i8 = hw.constant 0 : i8
  %false = hw.constant false
  %c0_i3 = hw.constant 0 : i3
  %rdptr0 = llhd.sig %c0_i3 : <i3>
  %rdptr0_dec = llhd.sig %c0_i8 : <i8>
  llhd.process {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb7
    %1 = llhd.prb %rdptr0 : i3
    %2 = comb.concat %c0_i5, %1 : i5, i3
    llhd.drv %rdptr0_dec, %2 after %0 : i8
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb6
    cf.cond_br %false, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    %3 = llhd.prb %rdptr0 : i3
    %4 = comb.icmp eq %3, %c0_i3 : i3
    cf.cond_br %4, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    cf.br ^bb6
  ^bb5:  // pred: ^bb3
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    cf.br ^bb2
  ^bb7:  // pred: ^bb2
    llhd.wait (%c0_i3, %c0_i8, %c0_i32, %false, %false, %c0_i3, %c0_i8, %c0_i8, %c0_i4, %c0_i4 : i3, i8, i32, i1, i1, i3, i8, i8, i4, i4), ^bb1
  }
  hw.output
}

// Probe an uninitialized signal after its declaration, even when it is only
// declared on one branch. Preserve the bits not overwritten by a partial drive.
// CHECK-LABEL: @UninitializedPartialDrive
hw.module @UninitializedPartialDrive() {
  %time = llhd.constant_time <0ns, 0d, 1e>
  %zero = hw.constant 0 : i12
  %offset = hw.constant 0 : i5
  %false = hw.constant false
  llhd.process {
    cf.cond_br %false, ^bb1, ^bb2
  ^bb1:
    // CHECK: ^bb1:
    // CHECK-NEXT: %[[SIG:.+]] = llhd.sig : <i32>
    %sig = llhd.sig : <i32>
    // CHECK-NEXT: [[INIT:%.+]] = llhd.prb %[[SIG]] : i32
    // CHECK-NEXT: [[HIGH:%.+]] = comb.extract [[INIT]] from 12 : (i32) -> i20
    // CHECK-NEXT: [[VALUE:%.+]] = comb.concat [[HIGH]], %c0_i12 : i20, i12
    %part = llhd.sig.extract %sig from %offset : <i32> -> <i12>
    llhd.drv %part, %zero after %time : i12
    // CHECK-NEXT: call @use_i32([[VALUE]])
    %value = llhd.prb %sig : i32
    func.call @use_i32(%value) : (i32) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %[[SIG]], [[VALUE]]
    cf.br ^bb2
  ^bb2:
    llhd.halt
  }
}

// A full write supplies the value needed by a subsequent conditional write,
// even if the signal has no initializer. No probe is needed.
// CHECK-LABEL: @UninitializedWrittenBeforeConditionalDrive
hw.module @UninitializedWrittenBeforeConditionalDrive(in %u: i42, in %v: i42, in %q: i1) {
  %time = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[VALUE:%.+]] = comb.mux %q, %v, %u : i42
    llhd.drv %sig, %u after %time : i42
    llhd.drv %sig, %v after %time if %q : i42
    %value = llhd.prb %sig : i42
    // CHECK-NEXT: call @use_i42([[VALUE]])
    func.call @use_i42(%value) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %sig, [[VALUE]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// A full write also supplies the value for a subsequent partial write. The
// local signal can be removed without probing its unspecified initial value.
// CHECK-LABEL: @UninitializedWrittenBeforePartialDrive
hw.module @UninitializedWrittenBeforePartialDrive(in %u: !hw.array<4xi42>, in %v: i42, in %i: i2) {
  %time = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.process
  llhd.process {
    %sig = llhd.sig : <!hw.array<4xi42>>
    llhd.drv %sig, %u after %time : !hw.array<4xi42>
    %part = llhd.sig.array_get %sig[%i] : <!hw.array<4xi42>>
    // CHECK-NEXT: [[VALUE:%.+]] = hw.array_inject %u[%i], %v
    llhd.drv %part, %v after %time : i42
    %value = llhd.prb %sig : !hw.array<4xi42>
    // CHECK-NEXT: call @use_array_i42([[VALUE]])
    func.call @use_array_i42(%value) : (!hw.array<4xi42>) -> ()
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Reentering the declaration starts a new signal lifetime. Each iteration
// must probe the new signal for the value forwarded when the drive is disabled.
// CHECK-LABEL: @UninitializedConditionalDriveInLoop
hw.module @UninitializedConditionalDriveInLoop(in %v: i42, in %q: i1, in %again: i1) {
  %time = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: cf.br ^bb1
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: %sig = llhd.sig : <i42>
    %sig = llhd.sig : <i42>
    // CHECK-NEXT: [[INIT:%.+]] = llhd.prb %sig : i42
    // CHECK-NEXT: [[VALUE:%.+]] = comb.mux %q, %v, [[INIT]] : i42
    llhd.drv %sig, %v after %time if %q : i42
    %value = llhd.prb %sig : i42
    // CHECK-NEXT: call @use_i42([[VALUE]])
    func.call @use_i42(%value) : (i42) -> ()
    cf.cond_br %again, ^bb1, ^bb2
  ^bb2:
    llhd.halt
  }
}

// A delayed partial drive updates the pending value, while probes still see
// the starting value. Both values need the read after the local declaration.
// CHECK-LABEL: @UninitializedDelayedPartialDrive
hw.module @UninitializedDelayedPartialDrive(in %v: i42, in %i: i2) {
  %time = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: %sig = llhd.sig : <!hw.array<4xi42>>
    %sig = llhd.sig : <!hw.array<4xi42>>
    // CHECK-NEXT: [[INIT:%.+]] = llhd.prb %sig
    %part = llhd.sig.array_get %sig[%i] : <!hw.array<4xi42>>
    // CHECK-NEXT: [[VALUE:%.+]] = hw.array_inject [[INIT]][%i], %v
    llhd.drv %part, %v after %time : i42
    %value = llhd.prb %sig : !hw.array<4xi42>
    // CHECK-NEXT: call @use_array_i42([[INIT]])
    func.call @use_array_i42(%value) : (!hw.array<4xi42>) -> ()
    // CHECK-NEXT: [[TIME:%.+]] = llhd.constant_time <0ns, 1d, 0e>
    // CHECK-NEXT: llhd.drv %sig, [[VALUE]] after [[TIME]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// An explicit initializer likewise supplies the starting value for delayed
// partial writes, so a read of the signal can be forwarded to the initializer.
// Removing the dead signal and drives must preserve unused module arguments.
// CHECK-LABEL: @InitializedDelayedPartialDrive
// CHECK-SAME: (in %u : !hw.array<4xi42>, in %v : i42, in %i : i2)
hw.module @InitializedDelayedPartialDrive(in %u: !hw.array<4xi42>, in %v: i42, in %i: i2) {
  %time = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.process
  llhd.process {
    %sig = llhd.sig %u : <!hw.array<4xi42>>
    %part = llhd.sig.array_get %sig[%i] : <!hw.array<4xi42>>
    llhd.drv %part, %v after %time : i42
    %value = llhd.prb %sig : !hw.array<4xi42>
    // CHECK-NEXT: call @use_array_i42(%u)
    func.call @use_array_i42(%value) : (!hw.array<4xi42>) -> ()
     // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Projection chains of slots, including any constants defined inside the
// process, are moved out of the process.
// CHECK-LABEL: @HoistSlotWithLocalConstants
hw.module @HoistSlotWithLocalConstants(in %u: i1) {
  %c0_i8 = hw.constant 0 : i8
  %init = hw.array_create %c0_i8, %c0_i8, %c0_i8, %c0_i8 : i8
  // CHECK: [[A:%.+]] = llhd.sig
  %a = llhd.sig %init : <!hw.array<4xi8>>
  // CHECK-NEXT: [[C1_I2:%.+]] = hw.constant 1 : i2
  // CHECK-NEXT: [[ELEM:%.+]] = llhd.sig.array_get [[A]][[[C1_I2]]]
  // CHECK-NEXT: [[C1_I3:%.+]] = hw.constant 1 : i3
  // CHECK-NEXT: [[BIT:%.+]] = llhd.sig.extract [[ELEM]] from [[C1_I3]]
  // CHECK-NEXT: llhd.process {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %c1_i2 = hw.constant 1 : i2
    %c1_i3 = hw.constant 1 : i3
    %0 = llhd.sig.array_get %a[%c1_i2] : <!hw.array<4xi8>>
    %1 = llhd.sig.extract %0 from %c1_i3 : <i8> -> <i1>
    // CHECK: llhd.drv [[BIT]], %u
    llhd.drv %1, %u after %eps : i1
    llhd.wait ^bb1
  }
}

// Constant projections into provably disjoint parts of a signal are promoted
// as independent slots. Each projection becomes the slot and is hoisted out of
// the process, and the parent signal is never probed or driven as a whole.
// CHECK-LABEL: @DisjointProjectionsPromotedIndependently
hw.module @DisjointProjectionsPromotedIndependently(in %u: !hw.array<4xi8>, in %v: i8, in %w: i8) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  // CHECK: %a = llhd.sig
  %a = llhd.sig %u : <!hw.array<4xi8>>
  // CHECK-DAG: [[A0:%.+]] = llhd.sig.array_get %a[%c0_i2]
  // CHECK-DAG: [[A1:%.+]] = llhd.sig.array_get %a[%c1_i2]
  // CHECK: llhd.process {
  llhd.process {
    // CHECK-NOT: llhd.prb
    // CHECK-NOT: llhd.sig.array_get
    %1 = llhd.sig.array_get %a[%c0_i2] : <!hw.array<4xi8>>
    %2 = llhd.sig.array_get %a[%c1_i2] : <!hw.array<4xi8>>
    llhd.drv %1, %v after %0 : i8
    llhd.drv %2, %w after %0 : i8
    %3 = llhd.prb %1 : i8
    %4 = llhd.prb %2 : i8
    // CHECK-NEXT: call @use_i8(%v)
    // CHECK-NEXT: call @use_i8(%w)
    func.call @use_i8(%3) : (i8) -> ()
    func.call @use_i8(%4) : (i8) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-DAG: llhd.drv [[A0]], %v
    // CHECK-DAG: llhd.drv [[A1]], %w
    // CHECK-NOT: llhd.drv %a,
    // CHECK: llhd.halt
    llhd.halt
  }
}

// A dynamic projection can alias any of its siblings, so the parent signal is
// promoted as a whole even though the other projection has a constant index.
// CHECK-LABEL: @DynamicProjectionPromotesParent
hw.module @DynamicProjectionPromotesParent(in %u: !hw.array<4xi8>, in %v: i8, in %w: i8, in %i: i2) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i2 = hw.constant 0 : i2
  %a = llhd.sig %u : <!hw.array<4xi8>>
  // CHECK-NOT: llhd.sig.array_get
  // CHECK: llhd.process {
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NOT: llhd.sig.array_get
    %1 = llhd.sig.array_get %a[%c0_i2] : <!hw.array<4xi8>>
    %2 = llhd.sig.array_get %a[%i] : <!hw.array<4xi8>>
    // CHECK-NEXT: [[INJ1:%.+]] = hw.array_inject [[A]][%c0_i2], %v
    llhd.drv %1, %v after %0 : i8
    // CHECK-NEXT: [[INJ2:%.+]] = hw.array_inject [[INJ1]][%i], %w
    llhd.drv %2, %w after %0 : i8
    // The probe must observe the dynamic drive, which may have hit index 0.
    // CHECK-NEXT: [[GET:%.+]] = hw.array_get [[INJ2]][%c0_i2]
    %3 = llhd.prb %1 : i8
    // CHECK-NEXT: call @use_i8([[GET]])
    func.call @use_i8(%3) : (i8) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[INJ2]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Constant projections whose ranges overlap are not independent, so the parent
// signal is promoted as a whole.
// CHECK-LABEL: @OverlappingProjectionsPromoteParent
hw.module @OverlappingProjectionsPromoteParent(in %u: i16, in %v: i8, in %w: i8) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i4 = hw.constant 0 : i4
  %c4_i4 = hw.constant 4 : i4
  %a = llhd.sig %u : <i16>
  // CHECK-NOT: llhd.sig.extract
  // CHECK: llhd.process {
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NOT: llhd.sig.extract
    // Bits [0, 8) and [4, 12) overlap in [4, 8).
    %1 = llhd.sig.extract %a from %c0_i4 : <i16> -> <i8>
    %2 = llhd.sig.extract %a from %c4_i4 : <i16> -> <i8>
    // CHECK-NEXT: [[HI:%.+]] = comb.extract [[A]] from 8
    // CHECK-NEXT: [[DRV1:%.+]] = comb.concat [[HI]], %v
    llhd.drv %1, %v after %0 : i8
    // CHECK-NEXT: [[HI:%.+]] = comb.extract [[DRV1]] from 12
    // CHECK-NEXT: [[LO:%.+]] = comb.extract [[DRV1]] from 0
    // CHECK-NEXT: [[DRV2:%.+]] = comb.concat [[HI]], %w, [[LO]]
    llhd.drv %2, %w after %0 : i8
    // CHECK-NEXT: [[PRB:%.+]] = comb.extract [[DRV2]] from 0
    %3 = llhd.prb %1 : i8
    // CHECK-NEXT: call @use_i8([[PRB]])
    func.call @use_i8(%3) : (i8) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[DRV2]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// A dynamic projection only forces its immediate parent to become the slot.
// Here `%a[0]` is promoted as a whole because of the dynamic `[%i]` below it,
// while the disjoint sibling `%a[1]` is still promoted independently.
// CHECK-LABEL: @NestedDynamicProjectionPromotesIntermediate
hw.module @NestedDynamicProjectionPromotesIntermediate(in %u: !hw.array<2xarray<4xi8>>, in %v: i8, in %w: !hw.array<4xi8>, in %i: i2) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %true = hw.constant true
  // CHECK: %a = llhd.sig
  %a = llhd.sig %u : <!hw.array<2xarray<4xi8>>>
  // CHECK-DAG: [[A0:%.+]] = llhd.sig.array_get %a[%false]
  // CHECK-DAG: [[A1:%.+]] = llhd.sig.array_get %a[%true]
  // CHECK: llhd.process {
  llhd.process {
    // CHECK-NEXT: [[ROW:%.+]] = llhd.prb [[A0]]
    // CHECK-NOT: llhd.sig.array_get
    %1 = llhd.sig.array_get %a[%false] : <!hw.array<2xarray<4xi8>>>
    %2 = llhd.sig.array_get %1[%i] : <!hw.array<4xi8>>
    %3 = llhd.sig.array_get %a[%true] : <!hw.array<2xarray<4xi8>>>
    // CHECK-NEXT: [[INJ:%.+]] = hw.array_inject [[ROW]][%i], %v
    llhd.drv %2, %v after %0 : i8
    llhd.drv %3, %w after %0 : !hw.array<4xi8>
    %4 = llhd.prb %2 : i8
    // CHECK-NEXT: call @use_i8(%v)
    func.call @use_i8(%4) : (i8) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-DAG: llhd.drv [[A0]], [[INJ]]
    // CHECK-DAG: llhd.drv [[A1]], %w
    // CHECK-NOT: llhd.drv %a,
    // CHECK: llhd.halt
    llhd.halt
  }
}

// Distinct struct fields occupy disjoint bit ranges and are promoted as
// independent slots.
// CHECK-LABEL: @DisjointStructFieldsPromotedIndependently
hw.module @DisjointStructFieldsPromotedIndependently(in %u: !hw.struct<x: i8, y: i8>, in %v: i8, in %w: i8) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: %a = llhd.sig
  %a = llhd.sig %u : <!hw.struct<x: i8, y: i8>>
  // CHECK-DAG: [[X:%.+]] = llhd.sig.struct_extract %a["x"]
  // CHECK-DAG: [[Y:%.+]] = llhd.sig.struct_extract %a["y"]
  // CHECK: llhd.process {
  llhd.process {
    // CHECK-NOT: llhd.prb
    // CHECK-NOT: llhd.sig.struct_extract
    %1 = llhd.sig.struct_extract %a["x"] : <!hw.struct<x: i8, y: i8>>
    %2 = llhd.sig.struct_extract %a["y"] : <!hw.struct<x: i8, y: i8>>
    llhd.drv %1, %v after %0 : i8
    llhd.drv %2, %w after %0 : i8
    %3 = llhd.prb %1 : i8
    %4 = llhd.prb %2 : i8
    // CHECK-NEXT: call @use_i8(%v)
    // CHECK-NEXT: call @use_i8(%w)
    func.call @use_i8(%3) : (i8) -> ()
    func.call @use_i8(%4) : (i8) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-DAG: llhd.drv [[X]], %v
    // CHECK-DAG: llhd.drv [[Y]], %w
    // CHECK-NOT: llhd.drv %a,
    // CHECK: llhd.halt
    llhd.halt
  }
}

// Union fields all alias the same storage, so the parent signal is promoted as
// a whole.
// CHECK-LABEL: @UnionFieldsPromoteParent
hw.module @UnionFieldsPromoteParent(in %u: !hw.union<x: i8, y: i8>, in %v: i8) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : <!hw.union<x: i8, y: i8>>
  // CHECK-NOT: llhd.sig.struct_extract
  // CHECK: llhd.process {
  llhd.process {
    // CHECK-NOT: llhd.sig.struct_extract
    %1 = llhd.sig.struct_extract %a["x"] : <!hw.union<x: i8, y: i8>>
    %2 = llhd.sig.struct_extract %a["y"] : <!hw.union<x: i8, y: i8>>
    // CHECK-NEXT: [[INJ:%.+]] = hw.union_create "x", %v
    llhd.drv %1, %v after %0 : i8
    // CHECK-NEXT: [[Y:%.+]] = hw.union_extract [[INJ]]["y"]
    %3 = llhd.prb %2 : i8
    // CHECK-NEXT: call @use_i8([[Y]])
    func.call @use_i8(%3) : (i8) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[INJ]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Disjoint projections do not help if the parent signal itself is accessed in
// the region: the parent is promoted as a whole.
// CHECK-LABEL: @DirectAccessPromotesParent
hw.module @DirectAccessPromotesParent(in %u: !hw.array<4xi8>, in %v: i8, in %w: i8) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %a = llhd.sig %u : <!hw.array<4xi8>>
  // CHECK-NOT: llhd.sig.array_get
  // CHECK: llhd.process {
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NOT: llhd.sig.array_get
    %1 = llhd.sig.array_get %a[%c0_i2] : <!hw.array<4xi8>>
    %2 = llhd.sig.array_get %a[%c1_i2] : <!hw.array<4xi8>>
    // CHECK-NEXT: [[INJ1:%.+]] = hw.array_inject [[A]][%c0_i2], %v
    llhd.drv %1, %v after %0 : i8
    // CHECK-NEXT: [[INJ2:%.+]] = hw.array_inject [[INJ1]][%c1_i2], %w
    llhd.drv %2, %w after %0 : i8
    %3 = llhd.prb %a : !hw.array<4xi8>
    // CHECK-NEXT: call @use_array_i8([[INJ2]])
    func.call @use_array_i8(%3) : (!hw.array<4xi8>) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[INJ2]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Blocking and delta drives to disjoint projections end up in separate slots,
// so the mixed-delay restriction (see @MultiDelayProjectionDrive) does not
// apply and both are promoted.
// CHECK-LABEL: @MixedDelaysOnDisjointSlots
hw.module @MixedDelaysOnDisjointSlots(in %v: i8, in %w: i8) {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %delta = llhd.constant_time <0ns, 1d, 0e>
  %false = hw.constant false
  %true = hw.constant true
  %init = hw.aggregate_constant [0 : i8, 0 : i8] : !hw.array<2xi8>
  // CHECK: %a = llhd.sig
  %a = llhd.sig %init : <!hw.array<2xi8>>
  // CHECK-DAG: [[A0:%.+]] = llhd.sig.array_get %a[%false]
  // CHECK-DAG: [[A1:%.+]] = llhd.sig.array_get %a[%true]
  // CHECK: llhd.process {
  llhd.process {
    // CHECK-NOT: llhd.prb
    // CHECK-NOT: llhd.sig.array_get
    %1 = llhd.sig.array_get %a[%false] : <!hw.array<2xi8>>
    %2 = llhd.sig.array_get %a[%true] : <!hw.array<2xi8>>
    llhd.drv %1, %v after %eps : i8
    llhd.drv %2, %w after %delta : i8
    %3 = llhd.prb %1 : i8
    // CHECK-NEXT: call @use_i8(%v)
    func.call @use_i8(%3) : (i8) -> ()
    // CHECK-DAG: llhd.drv [[A1]], %w after {{%.+}} : i8
    // CHECK-DAG: llhd.drv [[A0]], %v after {{%.+}} : i8
    // CHECK: llhd.halt
    llhd.halt
  }
}
