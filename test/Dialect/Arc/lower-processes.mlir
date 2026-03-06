// RUN: circt-opt %s --arc-lower-processes | FileCheck %s

// Test 1: Simple process with one wait and one halt
// CHECK-LABEL: arc.model @SimpleProcess
arc.model @SimpleProcess io !hw.modty<output x : i42> {
^bb0(%arg0: !arc.storage):
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c0_i42 = hw.constant 0 : i42
  %c42_i42 = hw.constant 42 : i42

  // CHECK: %[[RESUME_TIME_STATE:.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i64>
  // CHECK: %[[RESUME_BLOCK_STATE:.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i16>
  // CHECK: %[[PROC_RESULT_STATE:.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i42>
  // CHECK: %[[CURRENT_TIME:.+]] = arc.current_time %arg0
  // CHECK: %[[RESUME_TIME_READ:.+]] = arc.state_read %[[RESUME_TIME_STATE]]
  // CHECK: %[[SHOULD_RESUME:.+]] = comb.icmp uge %[[CURRENT_TIME]], %[[RESUME_TIME_READ]]
  // CHECK: scf.if %[[SHOULD_RESUME]] {
  // CHECK:   scf.execute_region {
  // CHECK:     %[[RESUME_BLOCK_READ:.+]] = arc.state_read %[[RESUME_BLOCK_STATE]]
  // CHECK:     cf.switch %[[RESUME_BLOCK_READ]] : i16, [
  // CHECK:       default: ^[[BB_HALTED:.+]],
  // CHECK:       0: ^[[BB_ENTRY:.+]],
  // CHECK:       1: ^[[BB_WAIT_TARGET:.+]]
  // CHECK:     ]
  // CHECK:   ^[[BB_ENTRY]]:
  // CHECK:     arc.state_write %[[PROC_RESULT_STATE]] = %c0_i42
  // CHECK:     arc.state_write %[[RESUME_TIME_STATE]] =
  // CHECK:     %[[C1:.+]] = arith.constant 1 : i16
  // CHECK:     arc.state_write %[[RESUME_BLOCK_STATE]] = %[[C1]]
  // CHECK:     scf.yield
  // CHECK:   ^[[BB_WAIT_TARGET]]:
  // CHECK:     arc.state_write %[[PROC_RESULT_STATE]] = %c42_i42
  // CHECK:     %[[C2:.+]] = arith.constant 2 : i16
  // CHECK:     arc.state_write %[[RESUME_BLOCK_STATE]] = %[[C2]]
  // CHECK:     scf.yield
  // CHECK:   ^[[BB_HALTED]]:
  // CHECK:     scf.yield
  // CHECK:   }
  // CHECK: }

  %1 = llhd.process -> i42 {
    llhd.wait yield (%c0_i42 : i42), delay %time, ^bb1
  ^bb1:
    llhd.halt %c42_i42 : i42
  }

  // CHECK: %[[PROC_RESULT:.+]] = arc.state_read %[[PROC_RESULT_STATE]]
}

// -----

// Test 2: Process with multiple waits (chain of waits)
// CHECK-LABEL: arc.model @MultipleWaits
arc.model @MultipleWaits io !hw.modty<output y : i8> {
^bb0(%arg0: !arc.storage):
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %c2_i8 = hw.constant 2 : i8
  %c3_i8 = hw.constant 3 : i8

  // CHECK: %[[RESUME_TIME_STATE:.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i64>
  // CHECK: %[[RESUME_BLOCK_STATE:.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i16>
  // CHECK: %[[PROC_RESULT_STATE:.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i8>

  // Verify the switch has 4 cases: entry (0), bb1 (1), bb2 (2), bb3 (3), plus halted (default)
  // CHECK: cf.switch %{{.+}} : i16, [
  // CHECK:   default: ^[[BB_HALTED:.+]],
  // CHECK:   0: ^[[BB_ENTRY:.+]],
  // CHECK:   1: ^[[BB1:.+]],
  // CHECK:   2: ^[[BB2:.+]],
  // CHECK:   3: ^[[BB3:.+]]
  // CHECK: ]

  // Entry block: wait to bb1
  // CHECK: ^[[BB_ENTRY]]:
  // CHECK:   arc.state_write %[[PROC_RESULT_STATE]] = %c0_i8
  // CHECK:   %[[C1:.+]] = arith.constant 1 : i16
  // CHECK:   arc.state_write %[[RESUME_BLOCK_STATE]] = %[[C1]]
  // CHECK:   scf.yield

  // bb1: wait to bb2
  // CHECK: ^[[BB1]]:
  // CHECK:   arc.state_write %[[PROC_RESULT_STATE]] = %c1_i8
  // CHECK:   %[[C2:.+]] = arith.constant 2 : i16
  // CHECK:   arc.state_write %[[RESUME_BLOCK_STATE]] = %[[C2]]
  // CHECK:   scf.yield

  // bb2: wait to bb3
  // CHECK: ^[[BB2]]:
  // CHECK:   arc.state_write %[[PROC_RESULT_STATE]] = %c2_i8
  // CHECK:   %[[C3:.+]] = arith.constant 3 : i16
  // CHECK:   arc.state_write %[[RESUME_BLOCK_STATE]] = %[[C3]]
  // CHECK:   scf.yield

  // bb3: halt
  // CHECK: ^[[BB3]]:
  // CHECK:   arc.state_write %[[PROC_RESULT_STATE]] = %c3_i8
  // CHECK:   %[[C4:.+]] = arith.constant 4 : i16
  // CHECK:   arc.state_write %[[RESUME_BLOCK_STATE]] = %[[C4]]
  // CHECK:   scf.yield

  %1 = llhd.process -> i8 {
    llhd.wait yield (%c0_i8 : i8), delay %time, ^bb1
  ^bb1:
    llhd.wait yield (%c1_i8 : i8), delay %time, ^bb2
  ^bb2:
    llhd.wait yield (%c2_i8 : i8), delay %time, ^bb3
  ^bb3:
    llhd.halt %c3_i8 : i8
  }

  // CHECK: arc.state_read %[[PROC_RESULT_STATE]]
}

// -----

// Test 3: Process with operations in blocks (not just terminators)
// CHECK-LABEL: arc.model @ProcessWithOps
arc.model @ProcessWithOps io !hw.modty<output z : i32> {
^bb0(%arg0: !arc.storage):
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c10_i32 = hw.constant 10 : i32
  %c20_i32 = hw.constant 20 : i32

  // CHECK: %[[RESUME_TIME_STATE:.+]] = arc.alloc_state %arg0
  // CHECK: %[[RESUME_BLOCK_STATE:.+]] = arc.alloc_state %arg0
  // CHECK: %[[PROC_RESULT_STATE:.+]] = arc.alloc_state %arg0

  // CHECK: cf.switch %{{.+}} : i16, [
  // CHECK:   default: ^[[BB_HALTED:.+]],
  // CHECK:   0: ^[[BB_ENTRY:.+]],
  // CHECK:   1: ^[[BB1:.+]]
  // CHECK: ]

  // Entry block should have the add operation cloned
  // CHECK: ^[[BB_ENTRY]]:
  // CHECK:   %[[ADD:.+]] = comb.add %c10_i32, %c20_i32
  // CHECK:   arc.state_write %[[PROC_RESULT_STATE]] = %[[ADD]]
  // CHECK:   %[[C1:.+]] = arith.constant 1 : i16
  // CHECK:   arc.state_write %[[RESUME_BLOCK_STATE]] = %[[C1]]
  // CHECK:   scf.yield

  // bb1 should have the mul operation cloned
  // CHECK: ^[[BB1]]:
  // CHECK:   %[[MUL:.+]] = comb.mul %c10_i32, %c20_i32
  // CHECK:   arc.state_write %[[PROC_RESULT_STATE]] = %[[MUL]]
  // CHECK:   %[[C2:.+]] = arith.constant 2 : i16
  // CHECK:   arc.state_write %[[RESUME_BLOCK_STATE]] = %[[C2]]
  // CHECK:   scf.yield

  %1 = llhd.process -> i32 {
    %add = comb.add %c10_i32, %c20_i32 : i32
    llhd.wait yield (%add : i32), delay %time, ^bb1
  ^bb1:
    %mul = comb.mul %c10_i32, %c20_i32 : i32
    llhd.halt %mul : i32
  }

  // CHECK: arc.state_read %[[PROC_RESULT_STATE]]
}

// -----

// Test 4: Process without delay (immediate wait)
// CHECK-LABEL: arc.model @NoDelay
arc.model @NoDelay io !hw.modty<output w : i1> {
^bb0(%arg0: !arc.storage):
  %true = hw.constant true
  %false = hw.constant false

  // CHECK: arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i64>
  // CHECK: arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i16>
  // CHECK: arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1>

  // CHECK: cf.switch %{{.+}} : i16, [
  // CHECK:   default: ^[[BB_HALTED:.+]],
  // CHECK:   0: ^[[BB_ENTRY:.+]],
  // CHECK:   1: ^[[BB1:.+]]
  // CHECK: ]

  %1 = llhd.process -> i1 {
    llhd.wait yield (%true : i1), ^bb1
  ^bb1:
    llhd.halt %false : i1
  }

  // CHECK: arc.state_read
}

