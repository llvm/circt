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

func.func private @use_i42(%arg0: i42)
func.func private @use_inout_i42(%arg0: !hw.inout<i42>)
