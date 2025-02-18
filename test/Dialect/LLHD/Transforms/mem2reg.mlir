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
    // CHECK-NEXT: cf.cond_br
    cf.cond_br %bool, ^bb1, ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42(%u)
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: cf.br ^bb2
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

// Forwarding on a subset of control flow when drive does not dominate all probes.
// CHECK-LABEL: @IncompleteDefinitionOnSubset
hw.module @IncompleteDefinitionOnSubset(in %u: i42, in %bool: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NEXT: [[TMP:%.+]] = llhd.prb %a
    // CHECK-NEXT: cf.cond_br %bool, ^bb1, ^bb2([[TMP]] : i42)
    cf.cond_br %bool, ^bb1, ^bb2
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NOT llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: cf.br ^bb2(%u : i42)
    cf.br ^bb2
  ^bb2:
    // CHECK-NEXT: ^bb2([[TMP:%.+]]: i42):
    // CHECK-NOT: llhd.prb
    %1 = llhd.prb %a : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42([[TMP]])
    func.call @use_i42(%1) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[TMP]]
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
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[TMP]]
    // CHECK-NEXT: llhd.wait ^bb1
    llhd.wait ^bb1
  }
}

// Check that additional basic blocks get inserted to accomodate drives when
// control flow diverges and reaching definitions cannot continue into all
// successors.
// CHECK-LABEL: @InsertDriveBlocks
hw.module @InsertDriveBlocks(in %u: i42, in %bool: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NEXT: cf.cond_br %bool, ^bb1, ^bb3
    cf.cond_br %bool, ^bb1, ^bb3
  ^bb1:
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: llhd.halt
    llhd.halt
  ^bb2: // no predecessors
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: cf.br ^bb4
    cf.br ^bb3
    // Helper block inserted between ^bb0 and ^bb3:
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, %u
    // CHECK-NEXT: cf.br ^bb4
  ^bb3:
    // Drives cannot be inserted here, since not all control flow visits the
    // drives.
    // CHECK-NEXT: ^bb4:
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

func.func private @use_i42(%arg0: i42)
