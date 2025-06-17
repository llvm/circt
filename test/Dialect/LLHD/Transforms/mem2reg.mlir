// RUN: circt-opt --llhd-mem2reg %s | FileCheck %s

// Trivial drive forwarding.
// CHECK-LABEL: @Trivial
hw.module @Trivial(in %u: i42) {
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    %0 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42(%u)
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Drive forwarding across reconvergent control flow.
// CHECK-LABEL: @ReconvergentControlFlow
hw.module @ReconvergentControlFlow(in %u: i42, in %bool: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: cf.cond_br
    cf.cond_br %bool, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3
  ^bb2:
    cf.br ^bb3
  ^bb3:
    // CHECK: ^bb3:
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42(%u)
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: cf.br ^bb2(%u : i42)
    cf.br ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 : !hw.inout<i42>
    // CHECK-NOT: llhd.prb
    %2 = llhd.prb %a : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42(%v)
    func.call @use_i42(%2) : (i42) -> ()
    // CHECK-NEXT: cf.br ^bb2(%v : i42)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[TMP:%.+]]: i42):
    // CHECK-NOT: llhd.prb
    %3 = llhd.prb %a : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[UNDEF:%.+]] = hw.constant 0 : i42
    // CHECK-NEXT: [[FALSE:%.+]] = hw.constant false
    // CHECK-NEXT: cf.cond_br %bool, ^bb1, ^bb2([[UNDEF]], [[FALSE]] : i42, i1)
    cf.cond_br %bool, ^bb1, ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: %false = hw.constant false
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NEXT: cf.cond_br %bool, ^bb1, ^bb2([[A]], %false : i42, i1)
    cf.cond_br %bool, ^bb1, ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: cf.br ^bb2(%u, %true : i42, i1)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[A:%.+]]: i42, [[ACOND:%.+]]: i1):
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
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
    %0 = llhd.prb %a : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.prb %a
    llhd.prb %a : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  %b = llhd.sig %v : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    llhd.drv %b, %v after %0 : !hw.inout<i42>
    // CHECK-NEXT: cf.br ^bb2(%u, %v : i42, i42)
    cf.br ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 : !hw.inout<i42>
    llhd.drv %b, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: cf.br ^bb2(%v, %u : i42, i42)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[A:%.+]]: i42, [[B:%.+]]: i42):
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
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
  %x = llhd.sig %c0_i42 : i42
  %y = llhd.sig %c0_i42 : i42
  %z = llhd.sig %c0_i42 : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[TMP:%.+]] = llhd.prb %x
    // CHECK-NOT: llhd.prb
    // CHECK-NOT: llhd.drv
    %1 = llhd.prb %x : !hw.inout<i42>
    llhd.drv %y, %1 after %0 : !hw.inout<i42>
    %2 = llhd.prb %y : !hw.inout<i42>
    llhd.drv %z, %2 after %0 : !hw.inout<i42>
    %3 = llhd.prb %z : !hw.inout<i42>
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
  %a = llhd.sig %c0_i42 : i42
  %b = llhd.sig %c0_i42 : i42
  %c = llhd.sig %c0_i42 : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: %false = hw.constant false
    // CHECK-NEXT: [[B:%.+]] = llhd.prb %b
    // CHECK-NEXT: [[C:%.+]] = llhd.prb %c
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: cf.br ^bb2(%u, [[B]], %false, [[C]] : i42, i42, i1, i42)
    cf.br ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: [[C:%.+]] = llhd.prb %c
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 : !hw.inout<i42>
    llhd.drv %b, %v after %0 : !hw.inout<i42>
    // CHECK-NEXT: cf.br ^bb2(%v, %v, %true, [[C]] : i42, i42, i1, i42)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[A:%.+]]: i42, [[B:%.+]]: i42, [[BCOND:%.+]]: i1, [[C:%.+]]: i42):
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
    %2 = llhd.prb %b : !hw.inout<i42>
    %3 = llhd.prb %c : !hw.inout<i42>
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
  %a = llhd.sig %c0_i42 : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NEXT: cf.br ^bb1
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.prb
    %0 = llhd.prb %a : !hw.inout<i42>
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
  %a = llhd.sig %c0_i42 : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NEXT: cf.br ^bb1([[A]] : i42)
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: ^bb1([[A:%.+]]: i42):
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
    // CHECK-NEXT: [[ANEW:%.+]] = comb.add [[A]], %u
    %2 = comb.add %1, %u : i42
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %2 after %0 : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    func.call @use_inout_i42(%a) : (!hw.inout<i42>) -> ()
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    // CHECK-NOT: llhd.prb
    %0 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    %1 = llhd.prb %a : !hw.inout<i42>
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
  %a = llhd.sig %c0_i42 : i42
  %b = llhd.sig %c0_i42 : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    %1 = llhd.prb %a : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u after {{%.+}} if %q
    llhd.drv %a, %u after %0 if %q : !hw.inout<i42>
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 if %q : !hw.inout<i42>
    // CHECK-NEXT: cf.br ^bb2(%u : i42)
    cf.br ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 if %q : !hw.inout<i42>
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
    llhd.drv %a, %u after %0 if %q : !hw.inout<i42>
    // CHECK-NEXT: cf.br ^bb2(%u, %q : i42, i1)
    cf.br ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 if %r : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  // Conditional drives following non-conditional drives should create
  // multiplexers to modify the value forwarded as a reaching definition.
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %q, %v, %u : i42
    llhd.drv %a, %v after %0 if %q : !hw.inout<i42>
    // CHECK-NEXT: [[DRV2:%.+]] = comb.mux %r, %w, [[DRV1]] : i42
    llhd.drv %a, %w after %0 if %r : !hw.inout<i42>
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
    llhd.drv %a, %u after %0 if %q : !hw.inout<i42>
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %r, %v, %u : i42
    // CHECK-NEXT: [[ENABLE1:%.+]] = comb.or %r, %q : i1
    llhd.drv %a, %v after %0 if %r : !hw.inout<i42>
    // CHECK-NEXT: [[DRV2:%.+]] = comb.mux %s, %w, [[DRV1]] : i42
    // CHECK-NEXT: [[ENABLE2:%.+]] = comb.or %s, [[ENABLE1]] : i1
    llhd.drv %a, %w after %0 if %s : !hw.inout<i42>
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
    llhd.drv %a, %u after %0 if %q : !hw.inout<i42>
    // CHECK-NEXT: [[DRV2:%.+]] = comb.mux %r, %v, [[DRV1]] : i42
    // CHECK-NEXT: [[ENABLE2:%.+]] = comb.or %r, %q : i1
    llhd.drv %a, %v after %0 if %r : !hw.inout<i42>
    // CHECK-NEXT: [[DRV3:%.+]] = comb.mux %s, %w, [[DRV2]] : i42
    // CHECK-NEXT: [[ENABLE3:%.+]] = comb.or %s, [[ENABLE2]] : i1
    llhd.drv %a, %w after %0 if %s : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42([[DRV3]])
    %1 = llhd.prb %a : !hw.inout<i42>
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
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %q, %v, %u : i42
    llhd.drv %a, %v after %0 if %q : !hw.inout<i42>
    // CHECK-NEXT: [[DRV2:%.+]] = comb.mux %r, %w, [[DRV1]] : i42
    llhd.drv %a, %w after %0 if %r : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42([[DRV2]])
    %1 = llhd.prb %a : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  // Delayed drives after blocking drives persist.
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[T:%.+]] = llhd.constant_time <0ns, 0d, 1e>
    // CHECK-NEXT: llhd.drv %a, %u after [[T]]
    // CHECK-NEXT: [[T:%.+]] = llhd.constant_time <0ns, 1d, 0e>
    // CHECK-NEXT: llhd.drv %a, %v after [[T]]
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    llhd.drv %a, %v after %1 : !hw.inout<i42>
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // Later blocking drives erase earlier delayed drives.
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv %a, %u
    // CHECK-NEXT: [[T:%.+]] = llhd.constant_time <0ns, 0d, 1e>
    // CHECK-NEXT: llhd.drv %a, %v after [[T]]
    llhd.drv %a, %u after %1 : !hw.inout<i42>
    llhd.drv %a, %v after %0 : !hw.inout<i42>
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %1 : !hw.inout<i42>
    // CHECK-NEXT: hw.constant 0 : i42
    // CHECK-NEXT: hw.constant false
    // CHECK-NEXT: cf.cond_br %bool, ^bb1, ^bb2({{%c0_i42.*}}, {{%false.*}}, %u, {{%true.*}} : i42, i1, i42, i1)
    cf.cond_br %bool, ^bb1, ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %v after %0 : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %q, %v, %u : i42
    llhd.drv %a, %v after %0 if %q : !hw.inout<i42>
    // CHECK-NEXT: [[DRV2:%.+]] = comb.mux %r, %w, [[DRV1]] : i42
    llhd.drv %a, %w after %0 if %r : !hw.inout<i42>
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
  %a = llhd.sig %u : !hw.array<4xi42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.sig.array_get
    // CHECK-NOT: llhd.drv
    %1 = llhd.sig.array_get %a[%i] : !hw.inout<array<4xi42>>
    llhd.drv %a, %u after %0 : !hw.inout<array<4xi42>>
    // CHECK-NOT: llhd.prb
    // CHECK-NEXT: [[TMP:%.+]] = hw.array_get %u[%i]
    %2 = llhd.prb %1 : !hw.inout<i42>
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
  %a = llhd.sig %u : !hw.array<4xi42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<array<4xi42>>
    // CHECK-NOT: llhd.sig.array_get
    %1 = llhd.sig.array_get %a[%i] : !hw.inout<array<4xi42>>
    // CHECK-NOT: llhd.drv
    // CHECK-NEXT: [[A:%.+]] = hw.array_inject %u[%i], %v
    llhd.drv %1, %v after %0 : !hw.inout<i42>
    // CHECK-NOT: llhd.prb
    %2 = llhd.prb %a : !hw.inout<array<4xi42>>
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
  %a = llhd.sig %u : !hw.array<4xi42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.sig.array_get
    %1 = llhd.sig.array_get %a[%i] : !hw.inout<array<4xi42>>
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<array<4xi42>>
    // CHECK-NEXT: [[TMP:%.+]] = hw.array_get %u[%i]
    // CHECK-NEXT: [[FIELD1:%.+]] = comb.mux %q, %v, [[TMP]]
    // CHECK-NEXT: [[DRV1:%.+]] = hw.array_inject %u[%i], [[FIELD1]]
    llhd.drv %1, %v after %0 if %q : !hw.inout<i42>
    // CHECK-NEXT: [[FIELD2:%.+]] = comb.mux %r, %w, [[FIELD1]]
    // CHECK-NEXT: [[DRV2:%.+]] = hw.array_inject [[DRV1]][%i], [[FIELD2]]
    llhd.drv %1, %w after %0 if %r : !hw.inout<i42>
    // CHECK-NEXT: call @use_array_i42([[DRV2]])
    // CHECK-NEXT: call @use_i42([[FIELD2]])
    %2 = llhd.prb %a : !hw.inout<array<4xi42>>
    %3 = llhd.prb %1 : !hw.inout<i42>
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
    %1 = llhd.sig.array_get %a[%i] : !hw.inout<array<4xi42>>
    // CHECK-NEXT: [[DRV1:%.+]] = comb.mux %q, %u, [[A]]
    llhd.drv %a, %u after %0 if %q : !hw.inout<array<4xi42>>
    // CHECK-NEXT: [[TMP:%.+]] = hw.array_get [[DRV1]][%i]
    // CHECK-NEXT: [[FIELD2:%.+]] = comb.mux %r, %v, [[TMP]]
    // CHECK-NEXT: [[DRV2:%.+]] = hw.array_inject [[DRV1]][%i], [[FIELD2]]
    // CHECK-NEXT: [[ENABLE2:%.+]] = comb.or %r, %q
    llhd.drv %1, %v after %0 if %r : !hw.inout<i42>
    // CHECK-NEXT: [[FIELD3:%.+]] = comb.mux %s, %w, [[FIELD2]]
    // CHECK-NEXT: [[DRV3:%.+]] = hw.array_inject [[DRV2]][%i], [[FIELD3]]
    // CHECK-NEXT: [[ENABLE3:%.+]] = comb.or %s, [[ENABLE2]]
    llhd.drv %1, %w after %0 if %s : !hw.inout<i42>
    // CHECK-NEXT: call @use_array_i42([[DRV3]])
    // CHECK-NEXT: call @use_i42([[FIELD3]])
    %2 = llhd.prb %a : !hw.inout<array<4xi42>>
    %3 = llhd.prb %1 : !hw.inout<i42>
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
  %a = llhd.sig %u : !hw.array<4xi42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    %1 = llhd.sig.array_get %a[%i] : !hw.inout<array<4xi42>>
    // CHECK-NEXT: [[TMP:%.+]] = hw.array_inject [[A]][%i], %v
    llhd.drv %1, %v after %0 : !hw.inout<i42>
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[TMP]] after {{%.+}}
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @ProjectionThroughBlockArg
hw.module @ProjectionThroughBlockArg(in %u: !hw.array<4xi42>, in %v: i42, in %i: i2) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : !hw.array<4xi42>
  // CHECK: llhd.process
  llhd.process {
    %1 = llhd.sig.array_get %a[%i] : !hw.inout<array<4xi42>>
    // Pass projection through block argument
    cf.br ^bb1(%1 : !hw.inout<i42>)
  ^bb1(%2: !hw.inout<i42>):
    // CHECK: ^bb1([[ARG:%.+]]: !hw.inout<i42>):
    // CHECK-NEXT: llhd.prb [[ARG]]
    %3 = llhd.prb %2 : !hw.inout<i42>
    // CHECK-NEXT: llhd.drv [[ARG]]
    llhd.drv %2, %v after %0 : !hw.inout<i42>
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @MultipleArrayGetsSameIndex
hw.module @MultipleArrayGetsSameIndex(in %u: !hw.array<4xi42>, in %v: i42, in %w: i42, in %i: i2) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : !hw.array<4xi42>
  // CHECK: llhd.process
  llhd.process {
    // Two separate array_gets for the same index
    // CHECK-NOT: llhd.sig.array_get
    %get1 = llhd.sig.array_get %a[%i] : !hw.inout<array<4xi42>>
    %get2 = llhd.sig.array_get %a[%i] : !hw.inout<array<4xi42>>
    // Drive both projections with different values
    // CHECK-NOT: llhd.drv
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // CHECK-NEXT: [[DRV1:%.+]] = hw.array_inject [[A]][%i], %v
    // CHECK-NEXT: [[DRV2:%.+]] = hw.array_inject [[DRV1]][%i], %w
    llhd.drv %get1, %v after %0 : !hw.inout<i42>
    llhd.drv %get2, %w after %0 : !hw.inout<i42>
    // Probe both projections
    // CHECK-NOT: llhd.prb
    %prb1 = llhd.prb %get1 : !hw.inout<i42>
    %prb2 = llhd.prb %get2 : !hw.inout<i42>
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
  %a = llhd.sig %u : !hw.array<5xarray<6xarray<7xi42>>>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[A:%.+]] = llhd.prb %a
    // Three nested projections
    %get1 = llhd.sig.array_get %a[%i] : !hw.inout<array<5xarray<6xarray<7xi42>>>>
    %get2 = llhd.sig.array_get %get1[%j] : !hw.inout<array<6xarray<7xi42>>>
    %get3 = llhd.sig.array_get %get2[%k] : !hw.inout<array<7xi42>>
    // Drive the innermost projection
    // CHECK-NEXT: [[GET3:%.+]] = hw.array_get [[A]][%i]
    // CHECK-NEXT: [[GET2:%.+]] = hw.array_get [[GET3]][%j]
    // CHECK-NEXT: [[INJECT1:%.+]] = hw.array_inject [[GET2]][%k], %v
    // CHECK-NEXT: [[INJECT2:%.+]] = hw.array_inject [[GET3]][%j], [[INJECT1]]
    // CHECK-NEXT: [[INJECT3:%.+]] = hw.array_inject [[A]][%i], [[INJECT2]]
    llhd.drv %get3, %v after %0 : !hw.inout<i42>
    // Probe the innermost projection
    %prb = llhd.prb %get3 : !hw.inout<i42>
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
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NOT: llhd.sig.extract
    %1 = llhd.sig.extract %a from %i : (!hw.inout<i42>) -> !hw.inout<i10>
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
    llhd.drv %1, %v after %0 if %q : !hw.inout<i10>
    // CHECK-NOT: llhd.prb
    %2 = llhd.prb %a : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42([[INJ11]])
    func.call @use_i42(%2) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[INJ11]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// CHECK-LABEL: @CombCreateDynamicInject
hw.module @CombCreateDynamicInject(in %u: i42, in %v: i10, in %q: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : i42

  // offset = 0
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[TMP1:%.+]] = comb.extract %u from 10 : (i42) -> i32
    // CHECK-NEXT: [[TMP2:%.+]] = comb.concat [[TMP1]], %v : i32, i10
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[TMP2]]
    // CHECK-NEXT: llhd.halt
    %c0_i6 = hw.constant 0 : i6
    %1 = llhd.sig.extract %a from %c0_i6 : (!hw.inout<i42>) -> !hw.inout<i10>
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    llhd.drv %1, %v after %0 : !hw.inout<i10>
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
    %1 = llhd.sig.extract %a from %c20_i6 : (!hw.inout<i42>) -> !hw.inout<i10>
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    llhd.drv %1, %v after %0 : !hw.inout<i10>
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
    %1 = llhd.sig.extract %a from %c32_i6 : (!hw.inout<i42>) -> !hw.inout<i10>
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    llhd.drv %1, %v after %0 : !hw.inout<i10>
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
    %1 = llhd.sig.extract %a from %c37_i6 : (!hw.inout<i42>) -> !hw.inout<i10>
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    llhd.drv %1, %v after %0 : !hw.inout<i10>
    llhd.halt
  }

  // offset >= 42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: llhd.halt
    %c42_i6 = hw.constant 42 : i6
    %1 = llhd.sig.extract %a from %c42_i6 : (!hw.inout<i42>) -> !hw.inout<i10>
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    llhd.drv %1, %v after %0 : !hw.inout<i10>
    llhd.halt
  }
}

func.func private @use_i42(%arg0: i42)
func.func private @use_inout_i42(%arg0: !hw.inout<i42>)
func.func private @use_array_i42(%arg0: !hw.array<4xi42>)
